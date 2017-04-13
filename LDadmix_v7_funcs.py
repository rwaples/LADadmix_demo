
# coding: utf-8

# In[34]:

#from pyplink import PyPlink
import numpy as np
from numba import jit
import multiprocessing
import itertools

#import pandas as pd
#import itertools
#import scipy.spatial.distance
#import matplotlib.pyplot as plt
#import seaborn as sns
#import collections

#import msprime
#import allel

#from plinkio import plinkfile

#%matplotlib inline
#sns.set_style('white')
#from pylab import rcParams
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:90% !important; }</style>"))


# ### Simulate random haplotypes

# In[37]:

@jit(nopython=True)
def get_rand_hap_freqs(n=2):
    """returns an (n,4) dimensioned numpy array of random haplotype frequencies, 
    where n is the number of pops"""
    # have to remove the list comprehension
    res = np.zeros((n, 4))
    for i in range(n):
        res[i] = np.diff(np.concatenate((np.array([0]), np.sort(np.random.rand(3)), np.array([1.0]))))
    return(res)


# In[39]:

@jit(nopython=True)
def get_LL_numba(Q, H, code):
    """returns the likelihood of the genotype data given Q and H
    Q = admixture fractions
    H = estimates of source-specific haplotype frequencies"""
    
    ind_hap_freqs = np.dot(Q, H)
    LL = 0.0
    # had to index the tuples returned by np.where
    LL += np.log(    ind_hap_freqs[np.where(code == 0)[0], 0] * ind_hap_freqs[np.where(code == 0)[0], 0]).sum()
    LL += np.log(2 * ind_hap_freqs[np.where(code == 1)[0], 0] * ind_hap_freqs[np.where(code == 1)[0], 2]).sum()
    LL += np.log(    ind_hap_freqs[np.where(code == 2)[0], 2] * ind_hap_freqs[np.where(code == 2)[0], 2]).sum()
    LL += np.log(2 * ind_hap_freqs[np.where(code == 3)[0], 0] * ind_hap_freqs[np.where(code == 3)[0], 1]).sum()
    LL += np.log(2 * ind_hap_freqs[np.where(code == 4)[0], 0] * ind_hap_freqs[np.where(code == 4)[0], 3] 
               + 2 * ind_hap_freqs[np.where(code == 4)[0], 1] * ind_hap_freqs[np.where(code == 4)[0], 2]).sum()
    LL += np.log(2 * ind_hap_freqs[np.where(code == 5)[0], 2] * ind_hap_freqs[np.where(code == 5)[0], 3]).sum()
    LL += np.log(    ind_hap_freqs[np.where(code == 6)[0], 1] * ind_hap_freqs[np.where(code == 6)[0], 1]).sum()
    LL += np.log(2 * ind_hap_freqs[np.where(code == 7)[0], 1] * ind_hap_freqs[np.where(code == 7)[0], 3]).sum()
    LL += np.log(    ind_hap_freqs[np.where(code == 8)[0], 3] * ind_hap_freqs[np.where(code == 8)[0], 3]).sum()
    return(LL)


# # try multiprocessing

# pool = multiprocessing.Pool(processes = n)
# loci = 10
# inputs = itertools.combinations(geno_array[:loci], 2)
# pool_outputs = pool.map(get_r2, inputs, 10)
# pool.close() # no more tasks
# pool.join()
# pool_outputs

# In[75]:

@jit(nopython=True)
def do_multiEM(inputs):
    """"""
    H, Q, code = inputs # unpack the input
    max_iter = 40
    tol = 10e-6
    verbose = False 
        
    n_ind = len(Q)
    n_pops = Q.shape[1]
    G = np.array([0,3,1,4, 3,6,4,7, 1,4,2,5, 4,7,5,8]) # which combinations of haplotypes produce which genotypes
    #H = initial
    
    old_LL = get_LL_numba(Q=Q, H = H , code = code)
    #if verbose:
    #    print(old_LL)
    
    # start iters here
    for i in range(max_iter):
        norm = np.zeros(n_ind)
        isum = np.zeros((n_ind, n_pops, 4)) # hold sums over the haps from each pop in each ind
        for hap1 in range(4):                                # index of haplotype in first spot
            for hap2 in range(4):                            # index of haplotype in second spot
                for ind, icode in enumerate(code):   # individuals
                    if icode == G[4 * hap1 + hap2]:   # if the current pair of haplotypes is consistent with the given genotype
                        for z1 in range(n_pops):                     # source pop of hap1
                            for z2 in range(n_pops):                 # source pop of hap2 
                                raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]
                                isum[ind, z1, hap1] += raw
                                isum[ind, z2, hap2] += raw
                                norm[ind] += raw

        # normalized sum over individuals
        post = np.zeros((n_pops, 4))
        for ind in range(n_ind):
            for z in range(n_pops):
                for hap in range(4):
                    post[z, hap] += isum[ind, z, hap]/norm[ind]           
                    
        # below doesn't currently work with numba, making the loop above for post necessary
        #post = 2*(isum / isum.sum((1,2))[:, np.newaxis,np.newaxis]).sum(0) 

        # scale the sums so they sum to one  - now represents the haplotype frequencies within pops
        H = np.zeros((n_pops, 4))
        for p in range(n_pops):
            H[p] = post[p]/post[p].sum()
        
        # again numba doesn't like            
        #H = post/(post.sum(1)[:,np.newaxis])

        new_LL = get_LL_numba(Q=Q, H = H , code = code)
        delta_LL = new_LL - old_LL
        assert(delta_LL > 0)
        
        #if verbose:
        #    print(i, new_LL, delta_LL)
        
        if delta_LL < tol:
            break
        old_LL = new_LL
    
    return(H, new_LL, i)


# In[41]:

@jit(nopython=True)
def LDadmix(locus_1, locus_2, Q, tol = 1e-6, seed = 42):
    assert(len(locus_1) == len(locus_2))
    assert(len(locus_1) == len(Q))
    
    # deal with missing data here
    # deal with genotype values not in [0,1,2] here
    # deal with haplotype fixed in one population.
    # deal with random seed here?
    
    geno_codes = locus_1 + 3*locus_2 # unique value for each possible pair of genotypes

    n_pops = Q.shape[1]
    # initialize with random haplotype frequencies
    initial_hap_freqs = get_rand_hap_freqs(n=n_pops)
    
    estimates = do_EM_numba(initial = initial_hap_freqs, Q = Q, code = geno_codes, max_iter = 100, tol = tol)
    return(estimates)


# # Summary statistics

# In[42]:

@jit(nopython=True)
def get_LD_from_haplotype_freqs(freqs):
    """given a set of four haplotype frequencies x populations, returns r^2 and D in each pop"""
    # TODO add D'
    # 00, 01, 10, 11
    pA = freqs[:,2] + freqs[:,3]
    pB = freqs[:,1] + freqs[:,3]
    pAB= freqs[:,3]
    D = pAB - pA*pB
    r2 = (D**2)/(pA*pB*(1.0-pA)*(1.0-pB))
    return(r2, D)


# # Wrapper

# In[1]:

def wrap_LDadmix(genotypes, Q):
    #UNUSED
    """Runs LDadmix for each pair of loci. 
    Collects results in a data frame"""
    
    n_loci = len(genotypes)
    n_pairs = (n_loci * (n_loci - 1)) / 2
    n_pops = Q.shape[1]
    # empty data frame
    df = pd.DataFrame(columns=['locus_1', 'locus_2', 'pop', 'r2', 'D', 'H1', 'H2', 'H3', 'H4', 'LL', 'iters' ], index = range(n_pairs*n_pops))
    
    idx = 0
    for l1 in range(n_loci):
        for l2 in range(l1+1, n_loci):
            H, LL, iters = LDadmix(genotypes[l1], genotypes[l2], Q=Q)
            r2, D = get_LD_from_haplotype_freqs(H)
            for pop in range(n_pops):
                df.loc[idx] = pd.Series({'locus_1':l1, 'locus_2':l2, 'pop':pop, 'r2':r2[pop], 'D':D[pop], 
                                         'LL':LL, 'iters':iters, 'H1':H[pop,0], 'H2':H[pop,1], 'H3':H[pop,2], 'H4':H[pop,3]})
                idx +=1
    
    df[['locus_1', 'locus_2', 'pop', 'iters']] = df[['locus_1', 'locus_2', 'pop', 'iters']].astype(int)
    return(df)
    
    

