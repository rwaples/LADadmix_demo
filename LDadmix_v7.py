
# coding: utf-8

# In[1]:

import numpy as np
from numba import jit
import multiprocessing
import itertools
import pandas as pd
from plinkio import plinkfile
import argparse
import sys
import time


# In[2]:

import LDadmix_v7_funcs as LDadmix


# # Helper functions

# In[3]:

@jit(nopython=True)
def get_rand_hap_freqs(n=2):
    """returns an (n,4) dimensioned numpy array of random haplotype frequencies, 
    where n is the number of pops"""
    res = np.zeros((n, 4))
    for i in range(n):
        res[i] = np.diff(np.concatenate((np.array([0]), np.sort(np.random.rand(3)), np.array([1.0]))))
    return(res)


# In[4]:

@jit(nopython=True)
def get_geno_codes(genos):
    """turns each pair of genotypes into an integer from 0 to 8"""
    return(genos[0] + 3*genos[1])


# In[5]:

@jit(nopython=True)
def get_LD_from_haplotype_freqs(freqs):
    """given a set of four haplotype frequencies x populations, returns r^2 and D in each pop"""
    # 00, 01, 10, 11
    pA = freqs[:,2] + freqs[:,3]
    pB = freqs[:,1] + freqs[:,3]
    pAB= freqs[:,3]
    D = pAB - pA*pB
    # expected freqs
    pa = 1.0 - pA
    pb = 1.0 - pB
    pApB = pA*pB
    pApb = pA*pb
    papB = pa*pB
    papb = pa*pb
    A = np.minimum(pApb, papB) # Dmax when D is positive 
    B = np.minimum(pApB, papb) # Dmax when D is negative
    Dmax = np.where(D >= 0, A, B)
    Dprime = D/Dmax
    r2 = (D**2)/(pA*pB*pa*pb)
    return(r2, D, Dprime)


# In[6]:

def load_plinkfile(basepath):
    plink_file = plinkfile.open(basepath)
    sample_list = plink_file.get_samples()
    locus_list = plink_file.get_loci()
    my_array = np.zeros((len(plink_file.get_loci( )), len(plink_file.get_samples( ))))
    for i, el in enumerate(plink_file): 
        my_array[i] = el
    return(sample_list, locus_list, my_array.astype(np.int))


# In[7]:

parser = argparse.ArgumentParser()
parser.add_argument('-Q', type=str, default = None, help='path to Q file')
parser.add_argument('-G', type=str, default = './LDadmix.in.bed', help='path to plink bed file')
parser.add_argument('-O', type=str, default = './LDadmix.out',  help='path to output file')
parser.add_argument('-P', type=int, default=4, help='number of threads')
parser.add_argument('-L', type=int, default=50, help='analyze the first L loci')


# In[8]:

import __main__ as main
if hasattr(main, '__file__'): # if not interactive
    args = parser.parse_args()
else: # if interactive
    #prevents trying to parse the sys.argv[1] of the interactive session
    args = parser.parse_args(['-Q' './explore/prototype/example_1.admix.2.Q', 
                             '-G', './explore/prototype/example_1.ld'])

print("\n------------------\nParameters: ")
print("Q file: {}".format(args.Q))
print("Plink files: {}".format(args.G))
print("Output file: {}".format(args.O))
print("Number of threads: {}".format(args.P))
print("Max number of loci: {}".format(args.L))
print("------------------\n")


# # Load input data

# In[9]:

print("\n------------------\nLoading data:")

sample_list, locus_list, geno_array = load_plinkfile(args.G)
print("Shape of genotype data:\n\t{}\tloci\n\t{}\tindividuals".format(geno_array.shape[0], geno_array.shape[1]))

if args.Q is None:
    print("No Q matrix was specified, assuming a single population.")
    # Make a Q matrix with just one pop
    q = np.ones((geno_array.shape[1], 1))
else:
    q = pd.read_csv(args.Q, header = None, sep = ' ')
    q = q.values
print("Shape of Q data:\n\t{}\tindividuals\n\t{}\tpopulations".format(q.shape[0], q.shape[1]))


# In[10]:

# quick sanity checks
assert(q.shape[0] == geno_array.shape[1]), "The number of individuals in the Q file doesn't match the G file!"
assert(geno_array.shape[0] >= args.L), "You asked for more loci ({}) than are present in the G file ({})!".format(args.L, geno_array.shape[0])


# In[ ]:

print("Done loading data, starting LDadmix.")
print("------------------\n")


# In[21]:

nloci = args.L
npairs = (nloci*(nloci-1))/2
npops = q.shape[1]
print("\n------------------")
print("There are {} locus pairs to analyze.".format(npairs))

cpu = args.P
pool = multiprocessing.Pool(processes = cpu)
print("Using {} cpus".format(cpu))

start_time = time.time()

# make input iterators
Hs = itertools.imap(get_rand_hap_freqs, itertools.repeat(npops, npairs))
Qs = itertools.repeat(q)
codes = itertools.imap(get_geno_codes, itertools.combinations(geno_array[:nloci], 2))
inputs = itertools.izip(Hs, Qs, codes)

# do the calculations 
pool_outputs = pool.map(func = LDadmix.do_multiEM, iterable=inputs)
pool.close() # no more tasks
pool.join()

print('Done!')
print('*** Running time ***')
print("*** {:.2f} seconds ***".format(time.time() - start_time))
print("------------------\n ")


# In[19]:

print("\n------------------ ")
print("Writing results file: {}".format(args.O))
with open(args.O, 'w') as OUTFILE:
    # write header
    LDheader = ['r2_Pop{}'.format(x) for x in range(1, 1+ npops)] + ['D_Pop{}'.format(x) for x in range(1, 1+ npops)] + ['Dprime_Pop{}'.format(x) for x in range(1, 1+ npops)]
    hapheader = ['Hap{}_Pop{}'.format(hap, pop) for hap,pop in zip([1,2,3,4]*npops, [x for x in range(1, npops+1) for i in range(4)]
)]
    header = ['Locus1', 'Locus2'] + LDheader + hapheader +['LL', 'nIter']
    OUTFILE.write('\t'.join(header))
    OUTFILE.write('\n')

    for pair, res in zip(itertools.combinations(xrange(nloci), 2), pool_outputs):
        OUTFILE.write('{}\t{}'.format(pair[0], pair[1]))
        r2, D, Dprime = get_LD_from_haplotype_freqs(res[0])
        for xx in r2:
            OUTFILE.write('\t{}'.format(xx))
        for xx in D:
            OUTFILE.write('\t{}'.format(xx))
        for xx in Dprime:
            OUTFILE.write('\t{}'.format(xx))
        for xx in res[0].flatten():
            OUTFILE.write('\t{}'.format(xx))
        OUTFILE.write('\t{}'.format(res[1]))
        OUTFILE.write('\t{}'.format(res[2]))
        OUTFILE.write('\n')

print( "Done writing results file, exiting")
print("------------------\n ")


# In[ ]:



