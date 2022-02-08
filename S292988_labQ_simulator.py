import random
from queue import PriorityQueue
import math
import numpy as np
import scipy.stats
from scipy.stats import norm as norm
from scipy.stats import t as stud
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time


class Measure:
    def __init__(self, Narr=0, Ndep=0, NAveraegUser=0, OldTimeEvent=0, AverageDelay=0, Lost_c=0, Tot_failures=0, fail_time=0, totusers=0):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay
        self.Lost_c = Lost_c
        self.totfailures = Tot_failures
        self.fail_time = fail_time
        self.totusers = totusers


class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time


class queue_simulator():
    def __init__(self, load, service, sim_time, maxdelay=50, max_queue=4, failure_rate=0, repair_rate=0):
        self.load = load
        self.service = service
        self.sim_time = sim_time
        self.arrival = service/load
        self.users = 0
        self.measures = Measure()
        self.type1 = 1
        self.maxdelay = maxdelay
        self.max_queue = max_queue
        self.failure_rate = failure_rate
        self.repair_rate = repair_rate
        self.is_failure = False
        self.repair_time = 0

    def arrival_f(self, time, FES, queue):

        # cumulate statistics
        self.measures.arr += 1
        self.measures.oldT = time
        self.measures.ut += self.users*(time-self.measures.oldT)

        # sample the time until the next arrival
        inter_arrival = random.expovariate(lambd=1.0/self.arrival)

        # schedule the next arrival
        FES.put((time + inter_arrival, "arrival"))

        if self.users < self.max_queue:
            # create a record for the client
            client = Client(self.type1, time)
            # insert the record in the queue
            queue.append(client)
            self.users += 1
            self.measures.totusers += 1
        else:
            self.measures.Lost_c += 1

        # if the server is idle start the service
        if self.users == 1:
            # sample the service time
            service_time = random.expovariate(1.0/self.service)
            # schedule the departure of the client
            if self.is_failure == True:
                service_time += self.repair_time-time
            FES.put((time + service_time, "departure"))

    def departure(self, time, FES, queue):

        # get the first element from the queue
        client = queue.pop(0)

        # cumulate statistics
        self.measures.dep += 1
        self.measures.ut += self.users*(time-self.measures.oldT)
        self.measures.oldT = time
        self.measures.delay += (time-client.arrival_time)

        # update the state variable, by decreasing the no. of clients by 1
        self.users -= 1

        # check whether there are more clients to in the queue
        if self.users > 0:
            # sample the service time
            service_time = random.expovariate(1.0/self.service)
            # schedule the departure of the client
            if self.is_failure == True:
                service_time += self.repair_time-time
            FES.put((time + service_time, "departure"))

    def failure(self, time, FES):

        # cumulate statistics
        self.measures.oldT = time
        self.measures.totfailures += 1
        # sample the time until the next arrival
        time_to_repair = random.expovariate(lambd=self.repair_rate)
        self.measures.fail_time += time_to_repair
        self.repair_time = time_to_repair+time
        for i, el in enumerate(FES.queue):
            if el[1] == "departure":
                FES.queue[i] = (FES.queue[i][0] +
                                time_to_repair, FES.queue[i][1])
        # schedule the next arrival
        FES.put((time + time_to_repair, "repair"))

    def repair(self, time, FES):

        # cumulate statistics
        self.measures.oldT = time

        # sample the time until the next arrival
        time_to_fail = random.expovariate(lambd=self.failure_rate)

        # schedule the next arrival
        FES.put((time + time_to_fail, "failure"))

    def simulate(self):
        random.seed(42)
        np.random.seed(42)
        FES = PriorityQueue()
        FES.put((0, "arrival"))
        if (self.failure_rate > 0 and self.repair_rate > 0):
            FES.put((0+random.expovariate(lambd=self.failure_rate), "failure"))
        times = []
        queue = []
        countdelaybelow = 0
        timeold = 0
        timeidle = 0
        oldelay = 0
        time = 0

        while time < self.sim_time:
            times.append((self.users))
            (time, event_type) = FES.get()
            if event_type == "arrival":
                if self.users == 0:
                    timeidle += (time-timeold)
                self.arrival_f(time, FES, queue)
                oldelay = self.measures.delay

            elif event_type == "departure":
                self.departure(time, FES, queue)
                if self.measures.delay-oldelay < self.maxdelay:
                    countdelaybelow = countdelaybelow + 1
                if self.users == 0:
                    timeold = time

            elif event_type == "failure":
                self.failure(time, FES)
                self.is_failure = True

            elif event_type == "repair":
                self.repair(time, FES)
                self.is_failure = False

        averagedelay = self.measures.delay/self.measures.totusers
        delaybelowavg = countdelaybelow/self.measures.totusers
        newdata = {"load": self.load,
                   "service": self.service,
                   "simtime": self.sim_time,                   
                   "maxdelay": self.maxdelay,
                   "max_queue": self.max_queue,
                   "failure_rate": self.failure_rate,
                   "repair_rate": self.repair_rate,
                   "avg_service_time": self.measures.ut/self.measures.totusers,
                   "avg_interrarrival": self.measures.arr/self.sim_time,
                   "avg_delay": averagedelay,
                   "tot_arrivals": self.measures.arr,
                   "p_delay_below": delaybelowavg,
                   "p_idle": timeidle/time,
                   "p_lost_customer": self.measures.Lost_c/(self.measures.Lost_c+self.measures.totusers),
                   "users": self.measures.totusers,
                   "tot_failures": self.measures.totfailures,
                   "%_time_failure": self.measures.fail_time/self.sim_time}

        return newdata, times
