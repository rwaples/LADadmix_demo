
# coding: utf-8

# In[52]:

import numpy as np
from numba import jit
import multiprocessing
import itertools
import pandas as pd
from plinkio import plinkfile
import argparse
import sys
import time


# In[53]:

import LDadmix_v7_funcs as LDadmix


# # Helper functions

# In[ ]:

@jit(nopython=True)
def get_rand_hap_freqs(n=2):
    """returns an (n,4) dimensioned numpy array of random haplotype frequencies, 
    where n is the number of pops"""
    # have to remove the list comprehension
    res = np.zeros((n, 4))
    for i in range(n):
        res[i] = np.diff(np.concatenate((np.array([0]), np.sort(np.random.rand(3)), np.array([1.0]))))
    return(res)


# In[ ]:

#@jit(nopython=True)
def get_geno_codes(genos):
    return genos[0] + 3*genos[1]


# In[54]:

def get_LD_from_haplotype_freqs(freqs):
    """given a set of four haplotype frequencies x populations, returns r^2 and D in each pop"""
    # TODO add Dprime
    # 00, 01, 10, 11
    pA = freqs[:,2] + freqs[:,3]
    pB = freqs[:,1] + freqs[:,3]
    pAB= freqs[:,3]
    D = pAB - pA*pB
    r2 = (D**2)/(pA*pB*(1.0-pA)*(1.0-pB))
    return(r2, D)


# In[55]:

def load_plinkfile(basepath):
    plink_file = plinkfile.open(basepath)
    sample_list = plink_file.get_samples()
    locus_list = plink_file.get_loci()
    my_array = np.zeros((len(plink_file.get_loci( )), len(plink_file.get_samples( ))))
    for i, el in enumerate(plink_file): 
        my_array[i] = el
    return(sample_list, locus_list, my_array.astype(np.int))


# # Parse command line
# Arguments:
#     -Q : path to Q file
#     -G : path to G file
#     -P : number of threads
#     -O : path to output file
#     -L : max number of loci

# In[56]:

parser = argparse.ArgumentParser()
parser.add_argument('-Q', type=str, default = './LDadmix.in.Q', help='path to Q file')
parser.add_argument('-G', type=str, default = './LDadmix.in.bed', help='path to plink bed file')
parser.add_argument('-O', type=str, default = './LDadmix.out',  help='path to output file')
parser.add_argument('-P', type=int, default=4, help='number of threads')
parser.add_argument('-L', type=int, default=50, help='analyze the first L loci')


# In[ ]:




# In[59]:

if __name__ == "__main__":
    args = parser.parse_args()
    print("\n------------------\nParameters: ")
    print("Q file: {}".format(args.Q))
    print("Plink files: {}".format(args.G))
    print("Output file: {}".format(args.O))
    print("Number of threads: {}".format(args.P))
    print("Max number of loci: {}".format(args.L))
    print("------------------\n")


# # Load input data

# In[25]:

print("\n------------------\nLoading data ")

q = pd.read_csv(args.Q, header = None, sep = ' ')
q = q.values
print("Shape of Q array: {} individuals and {} populations".format(q.shape[0], q.shape[1]))


# In[ ]:

sample_list, locus_list, geno_array = load_plinkfile(args.G)


# In[ ]:

print("Shape of geno array: {} loci and {} individuals".format(geno_array.shape[0], geno_array.shape[1]))


# In[ ]:

# sanity checks
assert(q.shape[0] == geno_array.shape[1])
assert(geno_array.shape[0] >= args.L)


# In[63]:


args.L = 20


# In[64]:

print("Done loading genotype data, starting LDadmix.")
print("There are {} locus pairs to analyze.".format((args.L*(args.L-1))/2))
print("------------------\n")

start_time = time.time()


# In[ ]:

n = args.P
loci = args.L
pool = multiprocessing.Pool(processes = n)


# make input iterators
Hs = itertools.imap(get_rand_hap_freqs, itertools.repeat(2, (loci*(loci-1))/2))
Qs = itertools.repeat(q)
codes = itertools.imap(get_geno_codes, itertools.combinations(geno_array[:loci], 2))
inputs = itertools.izip(Hs, Qs, codes)

pool_outputs = pool.map(func = LDadmix.do_multiEM, iterable=inputs, chunksize = 100)
pool.close() # no more tasks
pool.join()
pool_outputs


# In[ ]:

print("\n------------------\nDone! ")
print('*** Execution time ***')
print("*** {} seconds ***" .format(time.time() - start_time))
print( "Writing results file: {}".format(args.O))


# In[ ]:

with open(args.O, 'w') as OUTFILE:
    npop = pool_outputs[0][0].shape[0]
    hapheader = ['Hap{}_Pop{}'.format(hap, pop) for hap,pop in zip([1,2,3,4]*npop, [x for x in range(1, npop+1) for i in range(4)]
)]
    header = ['Locus1', 'Locus2'] + hapheader +['LL', 'nIter']
    OUTFILE.write('\t'.join(header))
    OUTFILE.write('\n')

    for pair, res in zip(itertools.combinations(xrange(loci), 2), pool_outputs):
        OUTFILE.write('{}\t{}'.format(pair[0], pair[1]))
        for xx in res[0].flatten():
            OUTFILE.write('\t{}'.format(xx))
        OUTFILE.write('\t{}'.format(res[1]))
        OUTFILE.write('\t{}'.format(res[2]))
        OUTFILE.write('\n')

print( "Done writing results file, exiting")
quit()


# In[ ]:

assert False


# In[73]:

npop = 3
[]


# In[68]:

[[x]*4 for x in range(npop)]


# In[76]:

[x for x in range(1, npop+1) for i in range(4)]


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
    
    


# In[ ]:




# In[ ]:

assert False


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Prepare benchmarking

# In[47]:

data = geno_array[:2]
h = get_rand_hap_freqs(2)
codes = data[0] + 3*data[1]
q = q


# In[ ]:




# In[49]:

get_ipython().magic(u'time get_LL_numba(Q=q, H =h, code = codes)')


# In[ ]:




# # Start benchmarking

# In[53]:

get_ipython().magic(u'time do_EM_numba(code = codes, initial = h, Q=q)')


# In[237]:

get_ipython().magic(u'timeit do_EM_numba(code = codes, initial = h, Q=q)')


# In[ ]:




# In[60]:

get_ipython().magic(u'time wrap_LDadmix(genotypes=geno_array, Q=q)')


# In[ ]:




# In[ ]:




# In[76]:

def get_geno_codes(genos):
    return genos[0] + 3*genos[1]


# # Load the larger admix genotype array

# In[61]:


#import scipy.stats
#print multiprocessing.cpu_count()
n = 10


# In[87]:

sample_list, locus_list, geno_array = load_plinkfile('/home/ryan/LDadmix/explore/prototype/example_1.admix')
geno_array


# # Timings
# don't use a chunksize of 1, probably best left at default

# In[99]:

get_ipython().run_cell_magic(u'time', u'', u'pass\n\nn = 20\nloci = 500\npool = multiprocessing.Pool(processes = n)\n\n\n# make input iterators\nHs = itertools.imap(get_rand_hap_freqs, itertools.repeat(2, (loci*(loci-1))/2))\nQs = itertools.repeat(q)\ncodes = itertools.imap(get_geno_codes, itertools.combinations(geno_array[:loci], 2))\ninputs = itertools.izip(Hs, Qs, codes)\n\npool_outputs = pool.map(func = do_multiEM, iterable=inputs, chunksize = 100)\npool.close() # no more tasks\npool.join()\npool_outputs')


# In[100]:

get_ipython().run_cell_magic(u'time', u'', u'pass\n\nn = 10\nloci = 500\npool = multiprocessing.Pool(processes = n)\n\n\n# make input iterators\nHs = itertools.imap(get_rand_hap_freqs, itertools.repeat(2, (loci*(loci-1))/2))\nQs = itertools.repeat(q)\ncodes = itertools.imap(get_geno_codes, itertools.combinations(geno_array[:loci], 2))\ninputs = itertools.izip(Hs, Qs, codes)\n\npool_outputs = pool.map(func = do_multiEM, iterable=inputs, chunksize = 100)\npool.close() # no more tasks\npool.join()\npool_outputs')


# In[102]:

get_ipython().run_cell_magic(u'time', u'', u'pass\n\nn = 10\nloci = 500\npool = multiprocessing.Pool(processes = n)\n\n\n# make input iterators\nHs = itertools.imap(get_rand_hap_freqs, itertools.repeat(2, (loci*(loci-1))/2))\nQs = itertools.repeat(q)\ncodes = itertools.imap(get_geno_codes, itertools.combinations(geno_array[:loci], 2))\ninputs = itertools.izip(Hs, Qs, codes)\n\npool_outputs = pool.map(func = do_multiEM, iterable=inputs)\npool.close() # no more tasks\npool.join()\npool_outputs')


# In[105]:

get_ipython().run_cell_magic(u'time', u'', u'pass\n\nn = 20\nloci = 1000\npool = multiprocessing.Pool(processes = n)\n\n\n# make input iterators\nHs = itertools.imap(get_rand_hap_freqs, itertools.repeat(2, (loci*(loci-1))/2))\nQs = itertools.repeat(q)\ncodes = itertools.imap(get_geno_codes, itertools.combinations(geno_array[:loci], 2))\ninputs = itertools.izip(Hs, Qs, codes)\n\npool_outputs = pool.map(func = do_multiEM, iterable=inputs)\npool.close() # no more tasks\npool.join()\npool_outputs')


# In[106]:

print len(pool_outputs)


# In[107]:

pool_outputs[0]


# In[ ]:

assert False


# In[40]:

#rewrite the G array as a design matrix? 
G = np.array([0,3,1,4, 3,6,4,7, 1,4,2,5, 4,7,5,8])
Gsquare = G.reshape(4,4)
Gsquare


# In[42]:

design = np.zeros((4,4,9))
for h1 in range(4):
    for h2 in range(4):
         design[h1, h2, Gsquare[h1, h2]] = 1
design


# In[93]:

loci = 10
genocodes = np.array(list(itertools.imap(get_geno_codes, itertools.combinations(geno_array[:loci], 2))))
print codes.shape
genocodes[0]


# In[94]:

def get_geno_design(paircodes):
    res = np.zeros((len(paircodes), 9))
    for idx, code in enumerate(paircodes):
        res[idx, code] = 1
    return res


# In[645]:

g = get_geno_design(genocodes[0])
g.shape


# In[646]:

h = get_rand_hap_freqs(n=2)


# In[705]:

# want a nind*4*4 matrix showing which combination of haplotypes are cosisten with the genotypes
# use either of these
# maybe can sum over the 4,4? 
np.inner(g, design)
i = ind_possible_haps = np.tensordot(g, design, axes = (1, 2))
print i.shape

# try standardizing the values in each ind space in i
inorm = i/i.sum((1, 2))[:, np.newaxis, np.newaxis]
i[0:3]


# In[959]:

h


# In[721]:

(inorm * h[0] * h[0])[3]


# In[743]:

np.tensordot(q,h, axes = (1,0))[0]
# does this equal QxHx for each hap x?
# no idea, prob not
np.tensordot(q, np.stack([h,h]), axes = (1,0))[0]


# In[968]:

np.dot(q[:,0][:, np.newaxis], h[0,:][np.newaxis, :]).shape


# In[967]:

np.dot(q[:,1][:, np.newaxis], h[1,:][np.newaxis, :])


# In[954]:

(inorm * h[0]).shape
(h[0,:][np.newaxis, :])
(inorm * h[0]).shape


# In[955]:

np.dot(q[:,0][:, np.newaxis], h[0,:][np.newaxis, :]).shape


# In[ ]:

#  raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]


# In[978]:

# does this equal QxHx for each hap x?
# works for pops with index 0 and 1
pop0weight = np.dot(q[:,0][:, np.newaxis], h[0,:][np.newaxis, :])
pop1weight = np.dot(q[:,1][:, np.newaxis], h[1,:][np.newaxis, :])
popweights = np.stack([pop0weight, pop1weight], axis = 1) # ind * pop * hap
popweights.shape
# this doesnt accout for the haplotypes must be consistent with the genotype


# In[1062]:

pop1weight


# In[1036]:

h[0,:][np.newaxis, :]


# In[1140]:

# these do incorporate the consistent haplotypes? 
pop0z1consistent = (inorm * pop0weight[:, :, np.newaxis])
pop0z2consistent = (inorm * pop0weight[:, np.newaxis, :])
pop1z1consistent = (inorm * pop1weight[:, :, np.newaxis])
pop1z2consistent = (inorm * pop1weight[:, np.newaxis, :])
#i * popweights[:,0,:][:, :, np.newaxis]
print pop0z1consistent.shape
# pop0consistent is the pop0 haplotype freq by the pop0 admixture by the genotype consistent
pop0z1consistent[3]


# In[1141]:

# for each ind i 
    # for each chrom c
        # for each source pop z
            # for each hap h
                # gives the 
iczh = np.stack([pop0z1consistent, pop1z1consistent, pop0z2consistent, pop1z2consistent], axis = 1).reshape(400, 2, 2, 4, 4)
iczh.shape


# In[1142]:

print iczh[3]


# In[1139]:

# now to combine these weights in meaningful ways
# pop0source = (2*pop0consistent*pop0consistent + 2*pop0consistent*pop1consistent)
# pop1source = (2*pop1consistent*pop1consistent + 2*pop0consistent*pop1consistent)
# ind / chrom / pop / haps
iczh[:, 0, 0,:] * iczh[:, 1, 0,:] 


# In[1128]:

iczh.sum(0)


# In[ ]:




# In[1106]:

post_debug.sum()


# In[1075]:

pop0weight[0]


# In[1074]:

pop0z1consistent[0].sum(1)


# In[1055]:

# which zX should I use for each?
pop0z1consistent.sum(2)[3] # sum over rows
pop0z2consistent.sum(1)[3] # sum over columns


# In[1065]:

pop0z1consistent.shape


# In[ ]:




# In[ ]:




# In[1060]:

pop0weight[3]


# In[1059]:

pop0z1consistent[3]


# In[1056]:

pop0z2consistent[0]


# In[1057]:

pop0z1consistent[0]


# In[1038]:

pop0consistent[0]


# In[1026]:

(pop0consistent*pop0consistent)[0]


# In[1043]:

(i * pop0weight[:, np.newaxis, :])[0]


# In[1031]:

pop0source = (2*pop0consistent*pop0consistent + 2*pop0consistent*pop1consistent)
pop1source = (2*pop1consistent*pop1consistent + 2*pop0consistent*pop1consistent)
pop0source.sum((0, 1))
pop1source.sum((0, 1))


# In[1032]:

pop1source.sum((0, 2))


# In[1034]:

post_debug


# In[ ]:




# In[934]:

pop0sum = (2*pop0weight*pop0weight + 2*pop0weight*pop1weight)
pop1sum = (2*pop1weight*pop1weight + 2*pop1weight*pop0weight)
pop1sum.shape


# In[935]:

pop0sum


# In[936]:

pop1sum


# In[925]:

pop1sum[:,np.newaxis]


# In[927]:

(i*pop0sum +i*pop0sum[:,np.newaxis]).sum((1, 0))


# In[911]:

(i*pop1sum +i*pop1sum[:,np.newaxis]).sum((1, 0))


# In[ ]:




# In[884]:

post_debug


# In[801]:

q.shape, h.shape, np.stack([q, q]).shape


# In[802]:

np.tensordot(np.stack([q, q]), h, axes = (0,0))


# In[803]:

q[:,0]


# In[804]:

q.shape


# In[805]:

h.shape


# In[806]:

np.stack([h,h]).shape


# In[807]:

q[3,0]


# In[808]:

#  raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]


# In[809]:

((i * h[0])**2 + (i * h[0])*(i * h[1]))


# In[810]:

q.shape


# In[811]:

h[0]


# In[812]:

# try summing over the two haplotypes within i
# expected haplotype counts prior to q or h
icollapsed = i.sum(1) + i.sum(2)
print icollapsed.shape
icollapsed[0:13]


# In[813]:

q.shape


# In[814]:

q[:,0]


# In[815]:

np.tensordot(q,h, axes = [1,0])


# In[816]:

# make a p matrix representing the possible source pop combinations for each chromosome
# and combine with q
n_pops = 2
p = np.zeros((2, n_pops))
#p = np.array([[0,0], [0,1]])
p


# In[817]:

np.inner([0,1], [0,1], [0,1])


# In[818]:

np.tensordot(icollapsed, h, axes = [1,1]).shape


# In[819]:

#raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]


# In[820]:

#g.shape
i.shape


# In[821]:

q.shape


# In[822]:

h.shape


# In[823]:

np.tensordot(i, q, axes = (0,0)).shape


# In[824]:

i[0]


# In[825]:

np.dot(i[0], h.T )


# In[826]:

design.shape


# In[827]:

design.T.shape


# In[828]:

icollapsed


# In[829]:

i[0]


# In[830]:

# now we need to leverage information from the haplotpe frequencies in each pop (h) and the admixture fractions (q)


# In[831]:

errr = (np.tensordot(i, h, axes = [1,1]) + np.tensordot(i, h, axes = [2,1]))
errr.shape


# In[832]:

q.shape


# In[833]:

np.tensordot(errr, q, axes = [0,0]).sum(1).T.sum()


# In[834]:

post_debug


# In[835]:

#(np.tensordot(h, i, axes = [1,1]) + np.tensordot(h, i, axes = [1,2])).shape


# In[836]:

np.tensordot(h, i, axes = [1,1]).shape


# In[837]:

np.tensordot(h, i, axes = [1,1])


# In[838]:

q.shape


# In[839]:

qicollapsed = np.tensordot(q, icollapsed, axes = (0,0))
qicollapsed.shape


# In[840]:

h.shape


# In[841]:

qicollapsed.T * h.T


# In[842]:

np.tensordot(h, qicollapsed, axes = (0,0))


# In[843]:

post_debug


# In[844]:

q.shape


# In[ ]:




# In[845]:

#ih = np.tensordot(i, h, axes = (2, 1))
#ih.shape


# In[846]:

post_debug


# In[847]:

H_debug


# In[848]:

np.tensordot(ih, q, axes = (0,0)).shape


# In[849]:

qh = np.tensordot(q, h, axes=([1,0]))
qh.shape


# In[850]:

i.shape


# In[851]:

qhi = np.tensordot(qh, i, axes=([0,0]))
print qhi.shape
qhi.reshape(2,2,4,4)


# In[852]:

# or maybe join i and q
print i.shape, q.shape
# per pop contribution of haplotype combinations
iq = np.tensordot(q, i, axes = (0, 0))
print iq.shape
iq


# In[853]:

# how to think about this, sum over rows and columns?, 
iq.sum(1) + iq.sum(2)


# In[854]:

iq_sum = iq.sum(1) + iq.sum(2)
iq_sum.shape


# In[855]:

h


# In[856]:

np.tensordot(h, iq_sum)


# In[ ]:




# In[ ]:



