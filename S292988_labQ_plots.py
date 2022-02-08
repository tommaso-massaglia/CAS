
from S292988_labQ_simulator import queue_simulator
from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


loads = [0.3, 0.5, 0.7, 0.9]
service = [10, 20, 40, 50, 60]
sim_time = [6*1e5]
maxdelay = [15, 20, 30, 40]
max_queue = [4, 6, 8, 12, 15, float("inf")]
failure_rate = [0, 0.001, 0.005, 0.01]
repair_rate = [0.005, 0.008, 0.01]


res = []
for args in product(loads, service, sim_time, maxdelay, max_queue, failure_rate, repair_rate):
    qs = queue_simulator(*args)
    values = qs.simulate()[0]
    res.append(values)


res = pd.DataFrame(res)
res.to_csv("results.csv", index=False)


sns.set(font_scale=1.7)
sns.set_style("ticks")
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.heatmap(abs(res.corr()),cmap="YlGnBu")
plot.set(title="Correlation Heatmap")


# Load vs Avg_delay
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="avg_delay", ax=ax, ci=95,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, color="cornflowerblue")

# Load vs p_delay_below
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="p_delay_below", hue= "maxdelay", ax=ax, ci=95,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="tab10")

#Load vs p_idle
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="p_idle", ax=ax, ci=100,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="cornflowerblue")

#Load vs p_lost_customers
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="p_lost_customer", hue="max_queue", ax=ax, ci=95,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="tab10")

#Load vs avg_delay split by max queue
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res[res["max_queue"]!=float("inf")], x="load", y="avg_delay", hue="max_queue", ax=ax, ci=95,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="tab10")

#Load vs avg_service_time split by failure rate
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="avg_service_time", hue="failure_rate", ax=ax, ci=100,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="tab10")

#Load vs p_idle split by failure rate
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="p_idle", hue="failure_rate", ax=ax, ci=95,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="tab10")

#Load vs p_idle split by repair rate
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="p_idle", hue="repair_rate", ax=ax, ci=95,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="tab10")

#Load vs avg_service_time split by repair rate
plot, ax = plt.subplots(1,1, figsize=(10,8))
plot = sns.lineplot(data=res, x="load", y="avg_service_time", hue="repair_rate", ax=ax, ci=100,\
                                    marker="o", dashes=False, linewidth=6, markersize=16, palette="tab10")

#-----------------------------------------------------------------------
# This creates the density of users graph

loads = [0.3, 0.7, 0.9]
service = [10, 20, 40]
sim_time = [6*1e5]
maxdelay = [40]
max_queue = [4, 8, float("inf")]
failure_rate = [0, 0.001, 0.005]
repair_rate = [0.005]

sns.set(font_scale=1.7)
sns.set_style("ticks")
plot, ax = plt.subplots(1,1, figsize=(10,8))

for args in product(loads, service, sim_time, maxdelay, max_queue, failure_rate, repair_rate):
    qs = queue_simulator(*args)
    test = qs.simulate()[1]
    pd.Series(np.array(test)/np.linalg.norm(test)).plot(kind="kde", ax=ax, title="Overlapping of normalized user distributions")


