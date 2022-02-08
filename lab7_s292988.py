import pandas as pd
import numpy as np
import seaborn as sns
import random as rand
import math
import datetime
import matplotlib.pyplot as plt
import argparse

#Theoretical probability given n and m
def th_prob(m, n):
    return 1-math.exp(-m**2/(2*n))

#Class to save, check and generate birthdates
class population:
    def __init__(self, m, distribution):
        self.m = m
        self.distribution = distribution
        self.birthdatelist = []

    def simulate(self):
        for k in range(self.m):
            self.birthdatelist.append(np.random.choice(self.distribution["date"], p=self.distribution["prob"]))
            if self.contains_duplicates():
                break

    def get_birthdatelist(self):
        return self.birthdatelist

    def contains_duplicates(self):
        return int(len(np.unique(self.birthdatelist)) != len(self.birthdatelist))

#What runs the simulation, it runs three cycles, the first runs the simulation nexp times, the second iterates over the list of m's and the third one iterates over a specific m
#to check the probability for that m
def simulator(distr, nruns, popsizes, nexp):    
    res = []
    for j in range(nruns):
        for m in popsizes:
            count_dup = 0
            for k in range(nexp):
                pop = population(m, distr)
                pop.simulate()
                count_dup += pop.contains_duplicates()
            res.append((m, th_prob(m, len(distr)), count_dup/nruns))
    df_res = pd.DataFrame(res, columns = ["m", "th_prob", "dup_prob"])

    df_res["mape"] = abs(df_res["th_prob"]-df_res["dup_prob"])/(df_res["dup_prob"])
    df_res["mape"] = df_res["mape"].apply(lambda x: 0 if math.isnan(x)==1 or x == math.inf else x)
    mape = sum(df_res["mape"])/len(df_res)
    
    return df_res, mape

#The function to generate the plots
def plotter(df_res, mape):
    sns.set(font_scale=1.8)
    sns.set_style("ticks")
    plot, ax = plt.subplots(figsize=(12, 8))
    plot = sns.lineplot(data=df_res, x="m", y="dup_prob", ax=ax, ci=95, label ="Probability of duplicates",\
        marker="o", dashes=False, linewidth=6, markersize=15, markevery=2)
    plot = sns.lineplot(data=df_res, x="m", y="th_prob", ax=ax, label="Theoretical probability",\
        marker="^", dashes=False, linewidth=3, markersize=15, markevery=2)
    plot.set(xticks=np.arange(0,110,10), title=f"Accuracy: {round(1-mape,3)}")
    plt.show()

#Parser to get the commands while launching the script
parser = argparse.ArgumentParser()
parser.add_argument("-seed", type = int, required=False, help='Seed', default=42)
parser.add_argument("-distribution", type = str, required=False, help='The distribution of birthdates from a .csv with columns [date, n°births], if empty it\'s set as uniform', default="uniform")
parser.add_argument("-m", type = int, nargs="+", required=False, help='The list of m to test, can be passed as a list of space separated numbers', default=np.arange(0,100,5))
parser.add_argument("-nexp", type = int, required=False, help='How many experiments for each m', default=50)
parser.add_argument("-nruns", type = int, required=False, help='How many times to run the simulation', default=100)
args = parser.parse_args()

seed = args.seed
distribution = args.distribution
m = args.m
nexp = args.nexp
nruns = args.nruns

rand.seed(seed)
np.random.seed(seed)

#Probability with a uniform distribution of birthday

if distribution == "uniform":
    alldates = [x.strftime("%d-%b") for x in pd.date_range(datetime.date(2020, 1, 1), periods=365)]
    distr = pd.DataFrame(alldates, columns=["date"])
    distr["average"] = 1 
    distr["prob"] = (distr["average"]/distr["average"].sum())
    df_res, mape = simulator(distr, nruns, m, nexp)
    plotter(df_res, mape)

#Probability with a real distribution of birthday

else:
    distr = pd.read_csv(distribution)
    distr["prob"] = (distr["average"]/distr["average"].sum())
    df_res, mape = simulator(distr, nruns, m, nexp)
    plotter(df_res, mape)