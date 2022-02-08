import pandas as pd
import numpy as np
import seaborn as sns
import random as rand
import math
import datetime
import matplotlib.pyplot as plt
import argparse
from math import log, sqrt

#Generate the ms to test on
popsims = []
for el in [0.2, 0.4, 0.8, 1]:
    for el2 in [100, 1000, 10000, 100000, 1000000]:
        popsims.append(int(el*el2))

popsims = sorted(popsims)

#Class to store, run the simulation and check for collisions
class extension():
    def __init__(self, popsize, distr):
        self.popsize = popsize
        self.distr = distr
        self.birthdatelist = []
        self.colltime = 0
    
    def contains_collsion(self):
        if int(len(np.unique(self.birthdatelist)) != len(self.birthdatelist)):
            self.colltime+=1
            self.birthdatelist=[]
            return 1
        return 0
    
    def add_birth(self):
        self.birthdatelist.append(rand.choice(range(self.popsize)))
        
    def get_birthdate_list(self):
        return self.birthdatelist

#Theoretical probability as shown in the paper
def th_prob_ext(d):
    return math.ceil(sqrt(2*d*log(2))+(3-2*log(2))/(6)+(9-4*log(2)**2)/(72*sqrt(2*d*log(2)))-(2*log(2)**2)/(135*d))

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-seed", type = int, required=False, help='Seed', default=42)
parser.add_argument("-m", type = int, nargs="+", required=False, help='The list of m to test, can be passed as a list of space separated numbers', default=popsims)
parser.add_argument("-nexp", type = int, required=False, help='How many experiments for each m', default=100)
parser.add_argument("-nruns", type = int, required=False, help='How many times to run the simulation', default=5)
args = parser.parse_args()

seed = args.seed
popsim = args.m
nexp = args.nexp
nruns = args.nruns

peoplecoll = []
np.random.seed(seed)
rand.seed(seed)

#The simulation
for i in range(nruns):
    for k in popsim:
        sim = extension(k, "uniform")
        addedpeople_list = []
        collisions = 0
        for i in range(nexp):
            addedpeople = 0
            for n in range(k):
                sim.add_birth()
                addedpeople += 1
                if(sim.contains_collsion()):
                    collisions+=1
                    addedpeople_list.append(addedpeople)
                    #The simulation is stopped whenever a collision is found as we just care for a single one
                    break
            if collisions/nexp>0.5:
                peoplecoll.append((round(sum(addedpeople_list)/len(addedpeople_list),0), k, th_prob_ext(k)))
                break
    
df_res2 = pd.DataFrame(peoplecoll, columns=["npeople", "popsize", "thprob"])

#Compute the mean absolute percentage error
df_res2["mape"] = abs(df_res2["thprob"]-df_res2["npeople"])/(df_res2["npeople"])
df_res2["mape"] = df_res2["mape"].apply(lambda x: 0 if math.isnan(x)==1 or x == math.inf else x)
mape = sum(df_res2["mape"])/len(df_res2)

#Plotter
sns.set(font_scale=1.8)
sns.set_style("ticks")
plot, ax = plt.subplots(figsize=(12, 8))
plot = sns.lineplot(data=df_res2, x="popsize", y="npeople", ax=ax, ci=95, label ="Required people for 0.5 collision",\
    marker="o", dashes=False, linewidth=6, markersize=15)
plot = sns.lineplot(data=df_res2, x="popsize", y="thprob", ax=ax, label ="Theoretical Probability",\
    marker="o", dashes=False, linewidth=3, markersize=12)
plot.set(xscale="log", xticks=popsim, title=f"Accuracy: {round(1-mape,3)}")
plt.show()
