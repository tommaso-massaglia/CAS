import hashlib
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import gzip

parser = argparse.ArgumentParser()
parser.add_argument("--version", default="", type=str, required=False)
parser.add_argument("--savefig", default=False, type=bool, required=False)
args = parser.parse_args()

if args.version == "movies":
    with gzip.open("title.akas.tsv.gz", "r") as f:
        dataset = pd.read_csv(f, sep="\t")
    wordslist = dataset[dataset["region"]=="IT"]["title"]
else:
    wordslist = np.squeeze(pd.read_csv("https://raw.githubusercontent.com/napolux/paroleitaliane/master/paroleitaliane/660000_parole_italiane.txt", header=None)).values

def compute_hash(word, nbits):
    word_hash = hashlib.md5(word.encode('utf-8'))
    word_hash_int = int(word_hash.hexdigest(), 16)
    h = word_hash_int % 2**nbits
    return h

#Memory used by fingerprinting
def mem_fingerprint(m, bits):
    # Bits is the number of bits used to generate the fingerprint
    return (m*bits/8)/1024**2

#p_falsep by using fingerprinting
def p_falsep_fingerprint(m, bexp):
    return 1-(1-1/2**bexp)**m

#A simulator class is used to store the dictionary once and run the simulations more easily passing just the number of bits, the method outputs the p_falsep for that bit count
class simulator():
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def compute_hash(self, word, nbits):
        word_hash = hashlib.md5(word.encode('utf-8'))
        word_hash_int = int(word_hash.hexdigest(), 16)
        h = word_hash_int % 2**nbits
        return h
    
    def create_bit_string(self, dataset, nbits):
        temp_bit_string = np.zeros(2**nbits, dtype=np.int8)
        for el in dataset:
            temp_bit_string[self.compute_hash(el, nbits)] = 1
        return temp_bit_string    
        
    def simulate(self, nbits):
        bit_string = self.create_bit_string(self.dictionary, nbits)
        n_1bits = sum(bit_string)
                
        return n_1bits/2**nbits


#Run the simulation, save the output at each step
tres = []
sim = simulator(wordslist)
fullstr_mem = sum([len(word)+1 for word in wordslist])/1024**2
print(f"The memory required to store the dataset as strings is: {fullstr_mem:.2} MB")

for nbits in range(19, 27):
    tres.append((nbits, sim.simulate(nbits), p_falsep_fingerprint(len(wordslist), nbits), 2**nbits/8/1024**2, mem_fingerprint(len(wordslist),nbits)))
    
res = pd.DataFrame(tres, columns=["nbits","p_falsep_bitstr", "p_falsep_fingerp", "mem_bitstr", "mem_fingerp"])


print(res)

#This is just plotting

sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=res, x="nbits", y="p_falsep_bitstr", ax=ax, ci=95, label ="p_falsep_bitstr",\
    marker="o", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=res, x="nbits", y="p_falsep_fingerp", ax=ax, ci=95, label ="p_falsep_fingerprint",\
    marker="^", dashes=False, linewidth=3, markersize=10)
plot.set(ylabel="p_falsep", title=f'MAE between the two curves = {sum(abs(res["p_falsep_bitstr"]-res["p_falsep_fingerp"]))/len(res):.2e}')
if args.savefig:
    plt.savefig("p_falsep.png")


sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=res, x="nbits", y="mem_bitstr", ax=ax, ci=95, label ="mem_bitstr",\
    marker="o", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=res, x="nbits", y="mem_fingerp", ax=ax, ci=95, label ="mem_fingerp",\
    marker="^", dashes=False, linewidth=6, markersize=15)
plot.set(ylabel="memory_usage (in MB)",title=f'MAE between the two curves = {sum(abs(res["mem_bitstr"]-res["mem_fingerp"]))/len(res):.02}')
if args.savefig:
    plt.savefig("mem_usage.png")


