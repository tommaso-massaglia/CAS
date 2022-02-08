import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import gzip
import time
import hashlib


parser = argparse.ArgumentParser()
parser.add_argument("--version", default="", type=str, required=False)
parser.add_argument("--savefig", default=False, type=bool, required=False)
args = parser.parse_args()

#Function given in the assignment and slightly edited, computes k hashes with b bits for each word
def compute_all_hashes(md5, num_hashes, b):
    bits_to_update=[] # the list of bits to update is initially empty
    for i in range(num_hashes): # for each hash to generate
        value=md5 % (2 ** b) # take the last b bits for the hash value
        bits_to_update.append(value) # add the hash value in the list
        md5 = md5 // (2 ** 3) # right-shift the md5 by 3 bits
    return bits_to_update

def mem_fingerprint(m, bits):
    # Bits is the number of bits used to generate the fingerprint
    return (m*bits/8)/1024**2

#Function used to compute a single hash
def compute_hash(word, nbits):
    word_hash = hashlib.md5(word.encode('utf-8'))
    word_hash_int = int(word_hash.hexdigest(), 16)
    h = word_hash_int % 2**nbits
    return h

def p_falsep_fingerprint(m, b):
    return 1-(1-1/2**b)**m


if args.version == "movies":
    with gzip.open("title.akas.tsv.gz", "r") as f:
        dataset = pd.read_csv(f, sep="\t")
    wordslist = dataset[dataset["region"] == "IT"]["title"]
else:
    wordslist = np.squeeze(pd.read_csv(
        "https://raw.githubusercontent.com/napolux/paroleitaliane/master/paroleitaliane/660000_parole_italiane.txt", header=None)).values

# The simulator class containing all the methods required to compute the bloom filter, create a bit string and get the results needed 
# to generate the graphs

class simulator():
    def __init__(self, wordslist):
        self.wordslist = wordslist
        self.nwords = len(wordslist)
        
    #32 is the maximum number of hashes our bloom filter can compute, so past that we just force 32
    def optimal_hashes(self, nbits):
        num_hashes =  math.floor(2**nbits/self.nwords*math.log(2))
        if (nbits+3*num_hashes>128):
            return 32
        if num_hashes<1:
            return 1
        return num_hashes
        
    #Iterates over each words, compute which bits to update and do that
    def create_bloom_string(self, nbits, num_hashes):
        temp_bit_string = np.zeros(2**nbits, dtype=np.int8)
        for el in self.wordslist:
            word_hash = hashlib.md5(el.encode('utf-8'))
            word_hash_int = int(word_hash.hexdigest(), 16) # compute the hash
            all_bits_to_update=compute_all_hashes(word_hash_int, num_hashes, nbits)
            for el2 in all_bits_to_update:
                temp_bit_string[el2] = 1
        return temp_bit_string
    
    #Bit string hashing
    def create_bit_string(self, nbits):
        temp_bit_string = np.zeros(2**nbits, dtype=np.int8)
        for el in self.wordslist:
            temp_bit_string[compute_hash(el, nbits)] = 1
        return temp_bit_string    
    
    #Get the p_falsep out of bit string hashing
    def simulate_bitstr(self, nbits):
        bit_string = self.create_bit_string(nbits)
        n_1bits = sum(bit_string)
                
        return n_1bits/2**nbits
    
    #Get the p_falsep out of bloom filters
    def simulate_bloom(self, nbits, num_hashes = -1, return_n1bits = False):
        if num_hashes == -1: num_hashes = self.optimal_hashes(nbits)
        bit_string = self.create_bloom_string(nbits, num_hashes)
        n1_bits = sum(bit_string)   
        if return_n1bits == True: return n1_bits
        else:
            return (n1_bits/2**nbits)**num_hashes
    
    #Get the th p_falsep out of bloom filter
    def p_falsep_th_bloom(self,nbits, num_hashes = -1):
        if num_hashes == -1: num_hashes = self.optimal_hashes(nbits)
        return (1-math.e**((-num_hashes*self.nwords)/2**nbits))**num_hashes
    
    def mem_bitstr(self, nbits):
        return 2**nbits/8/1024**2
    
    #Get the estimated number of elements out of a bloom filter
    def estimated_n_elems(self, nbits, num_hashes = -1):
        if num_hashes == -1: num_hashes = self.optimal_hashes(nbits)
        return -((2**nbits/num_hashes)*math.log(1-(self.simulate_bloom(nbits, num_hashes, True)/2**nbits)))      


#Here the simulation is run, printing at each step the time it took
res = []
startime = time.time()
sim = simulator(wordslist)
fullstr_mem = sum([len(word)+1 for word in wordslist])/1024**2
print(
    f"The memory required to store the dataset as strings is: {fullstr_mem:.02} MB")

for nbits in range(19, 27):
    temp = {
        "nbits": nbits,
        "optimal_n_hash": sim.optimal_hashes(nbits),
        "p_falsep_bloom": sim.simulate_bloom(nbits),
        "p_falsep_bloom_th": sim.p_falsep_th_bloom(nbits),
        "p_falsep_bitstr": sim.simulate_bitstr(nbits),
        "p_falsep_fingerp": p_falsep_fingerprint(sim.nwords, nbits),
        "mem_bitstr": sim.mem_bitstr(nbits),
        "mem_fingerp": mem_fingerprint(sim.nwords, nbits)
    }
    print(f"Computatiuons for b={nbits} took {time.time()-startime:.02} s")
    startime = time.time()
    res.append(temp)
data = pd.DataFrame(res)
if args.savefig == True: data.to_csv("results.csv", index=False)

#Graphs printing and saving if required

sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=data, x="nbits", y="p_falsep_bloom", ax=ax, ci=95, label="p_falsep_bloom",
                    marker="o", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=data, x="nbits", y="p_falsep_bloom_th", ax=ax, ci=95, label="p_falsep_bloom_th",
                    marker="^", dashes=False, linewidth=3, markersize=10)
plot.set(ylabel="p_falsep",
         title=f'MAE between the two curves = {sum(abs(data["p_falsep_bloom"]-data["p_falsep_bloom_th"]))/len(data):.2e}')

if args.savefig == True:
    plt.savefig("p_falsep_bloom.png")


sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=data, x="nbits", y="p_falsep_bloom", ax=ax, ci=95, label="p_falsep_bloom",
                    marker="o", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=data, x="nbits", y="p_falsep_bitstr", ax=ax, ci=95, label="p_falsep_bitstr",
                    marker="s", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=data, x="nbits", y="p_falsep_fingerp", ax=ax, ci=95, label="p_falsep_fingerp",
                    marker="^", dashes=False, linewidth=6, markersize=15)
plot.set(ylabel="p_falsep")
if args.savefig == True:
    plt.savefig("p_falsep_all.png")


sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=data, x="nbits", y="mem_bitstr", ax=ax, ci=95, label="mem_bitstr/bloom",
                    marker="o", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=data, x="nbits", y="mem_fingerp", ax=ax, ci=95, label="mem_fingerp",
                    marker="s", dashes=False, linewidth=6, markersize=15)
plot.set(ylabel="Size (MB)")
if args.savefig == True:
    plt.savefig("mem.png")


#-------------------------------------------------------------
#Extension 1

sim = simulator(wordslist)
res_ext1 = []
for nbits in range(19, 27):
    optimal_hashes = sim.optimal_hashes(nbits)
    for i in range(1,33):
        temp = {
            "nbits": nbits,
            "optimal_hashes": optimal_hashes,
            "tested_hash": i,
            "p_falsep_bloom_o": sim.p_falsep_th_bloom(nbits),
            "p_falsep_bloom_t": sim.p_falsep_th_bloom(nbits, i)
        }
        res_ext1.append(temp)
data_ext1 = pd.DataFrame(res_ext1)

sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=data_ext1, x="nbits", y="p_falsep_bloom_o", ax=ax, ci=95, label ="p_falsep_bloom_optimum",\
    marker="o", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=data_ext1[data_ext1["tested_hash"]==32], x="nbits", y="p_falsep_bloom_t", ax=ax, ci=95, label ="p_falsep_bloom_32",\
    marker="^", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=data_ext1[data_ext1["tested_hash"]==1], x="nbits", y="p_falsep_bloom_t", ax=ax, ci=95, label ="p_falsep_bloom_1",\
    marker="^", dashes=False, linewidth=6, markersize=15)
plot.set(ylabel="p_falsep")

if args.savefig == True:
    plt.savefig("ext1.png")

#-------------------------------------------------------------
#Extension 2

res_ext2 = []

for i in np.arange(1.5, 4, 0.5):
    sim = simulator(wordslist[:int(len(wordslist)/i)])
    for nbits in range(19, 27):
        temp = {
            "nbits": nbits,
            "optimal_hashes": optimal_hashes,
            "estimated_1_bits": sim.estimated_n_elems(nbits),
            "actual_elems": sim.nwords
        }
        temp["delta"] = abs(temp["estimated_1_bits"]-temp["actual_elems"])
        res_ext2.append(temp)
data_ext2 = pd.DataFrame(res_ext2)


sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=data_ext2, x="actual_elems", y="delta", ax=ax, ci=95, label ="delta",\
    marker="o", dashes=False, linewidth=6, markersize=15)
plot.set(ylabel="n_elems", title=f'MAE between actual and estimated = {sum(abs(data_ext2["estimated_1_bits"]-data_ext2["actual_elems"]))/len(data)}')

if args.savefig == True:
    plt.savefig("ex2.png")
