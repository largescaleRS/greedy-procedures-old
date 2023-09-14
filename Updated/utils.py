
import numpy as np
import ray
import scipy.stats as st

import sys
if sys.platform =='win32':
    import time as time
    import win_precise_time as ptime
elif sys.platform =='darwin':
    import time as time
    import time as ptime


    
def parallel_experiments(rng,  generators, policy=None, remote_policy=None, delta=0.01, test_time=False, args={}):
    _start = time.time()
    results = []
    n_replications = len(generators)
    k = generators[0].syscount()
    print("--------New experiments with  k={}----------------------".format(k))
    for expe_id in range(n_replications):
        _seed = rng.integers(1, 10e8)
        if test_time:
            if expe_id == 0:
                start = time.time()
                policy(generators[expe_id],  **args)
                end = time.time()
                print("Single replication takes: {}s".format(end-start))
                print("Estimated serial total time: {}s".format((end-start)* n_replications))
        else:
            pass
        results.append(remote_policy.remote(generators[expe_id], seed=_seed, expe_id = expe_id, **args))
    print("Start to simulate... at {}".format(time.ctime()))
    results = ray.get(results)
    PCS, PGS = evaluate_PCS(generators, results), evaluate_PGS(generators, results, delta=delta)
    print("PCS:{}, PGS:{}".format(PCS, PGS))
    _end = time.time()
    print("Total time used: {}s, simulation ends at {}".format(_end-_start, time.ctime()))
    return PCS, PGS


class SCCVGenerator(object):
    def __init__(self, n_alternatives, gamma, var, best_index=0):
        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.zeros(n_alternatives)
        self.means[best_index] = gamma
        self.best_mean = gamma
        self.variances = np.ones(self.n_alternatives)*var
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        return self.n_alternatives
    
    
class EMCVGenerator(object):
    def __init__(self, n_alternatives, gamma, lamda, var, best_index=0):
        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.arange(n_alternatives)/n_alternatives*lamda
        self.means[best_index] = gamma
        self.best_mean = gamma
        self.variances = np.ones(self.n_alternatives)*var
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        return self.n_alternatives
    

class EMIVGenerator(object):
    def __init__(self, n_alternatives, gamma, lamda, varlow, varhigh, best_index=0):
        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.arange(n_alternatives)/n_alternatives*lamda
        self.means[best_index] = gamma
        self.best_mean = gamma
        
        self.variances = varlow + np.arange(n_alternatives)/n_alternatives*(varhigh-varlow)
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        return self.n_alternatives
    

class EMDVGenerator(object):
    def __init__(self, n_alternatives, gamma, lamda, varlow, varhigh, best_index=0):
        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.arange(n_alternatives)/n_alternatives*lamda
        self.means[best_index] = gamma
        self.best_mean = gamma
        
        self.variances = varhigh + np.arange(n_alternatives)/n_alternatives*(varlow-varhigh)
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        return self.n_alternatives

def maxG(k):
    _sum = 1
    G = 1
    while _sum < k:
        G += 1
        _sum  = G*(2**(G)-1)
    if _sum > k:
        G -= 1
    return G

def evaluate_PCS(generators, return_expe_best_ids):
        size = len(return_expe_best_ids)
        n_correct = 0
        for i in range(size):
            expe_id = return_expe_best_ids[i][0]
            estimated_bestid = return_expe_best_ids[i][1]
            if generators[expe_id].means[estimated_bestid]  == generators[expe_id].best_mean:
                n_correct += 1
        return n_correct/size

    
def evaluate_PGS(generators, return_expe_best_ids, delta=0.01):
        size = len(return_expe_best_ids)
        n_good = 0
        for i in range(size):
            expe_id = return_expe_best_ids[i][0]
            estimated_bestid = return_expe_best_ids[i][1]
            if generators[expe_id].best_mean - generators[expe_id].means[estimated_bestid]  < delta:
                n_good += 1
        return n_good/size 


def loadTrue(L):
    L1 = L
    L2 = L1
    total = int((L1 - 1) ** 2 * (L2-2)/2)
    means = np.loadtxt("results{}.csv".format(total), delimiter=",")
    count = 0
    slns = []
    for s1 in range(1, L1-1): # 1-18
        for s2 in range(1, L1-s1): # 1-18
            s3 = L1 - s1 - s2
            for s4 in range(1, L2): # 1-19
                s5 = L1 - s4
                sln = [s1, s2, s3, s4, s5]
                # true_values[sln] = results[count]
                slns.append(sln)
                count += 1
    slns = np.array(slns)
    # best_mean = np.max(means)
    # best_sln = slns[np.argmax(means)]
    # slns = slns[np.argsort(-means)]
    # means = -np.sort(-means)
    return slns, means


def loadTPMax(L1, L2):
    total = int((L1 - 1) ** 2 * (L2-2)/2)
    means = np.loadtxt("results_{}_{}.csv".format(L1, L2), delimiter=",")
    count = 0
    slns = []
    for s1 in range(1, L1-1): # 1-18
        for s2 in range(1, L1-s1): # 1-18
            s3 = L1 - s1 - s2
            for s4 in range(1, L2): # 1-19
                s5 = L2 - s4
                sln = [s1, s2, s3, s4, s5]
                # true_values[sln] = results[count]
                slns.append(sln)
                count += 1
    slns = np.array(slns)
    return slns, means


def TpMax(_njobs,  _nstages, _burnin, r, b, simulator):
    sTime1 = np.random.exponential(1/r[0], _njobs)
    sTime2 = np.random.exponential(1/r[1], _njobs)
    sTime3 = np.random.exponential(1/r[2], _njobs)
    b1 = b[0]
    b2 = b[1]
    return simulator(_njobs, _burnin, sTime1, sTime2, sTime3, b1, b2)


def evaluate_PCS_parallel(generator, results):
        size = len(results)
        n_correct = 0
        for i in range(size):
            if generator.means[results[i]]  == generator.best_mean:
                n_correct += 1
        return n_correct/size
    

def evaluate_PGS_parallel(generator, results, delta=0.01):
        size = len(results)
        n_good = 0
        for i in range(size):
            if generator.best_mean - generator.means[results[i]]  < delta:
                            n_good += 1 
        return n_good/size
    
    
class TpMaxGenerator(object):
    def __init__(self, _njobs,  _nstages, _burnin, slns, means, simulator):
        self._njobs = _njobs
        self._nstages = _nstages
        self._burnin = _burnin
        # suffle the orders
        ids = np.arange(len(slns))
        self.slns = slns
        self.means = means
        self.best_mean = np.max(self.means)
        self.simulator = simulator
        
    def get(self, index, n=1):
        r = self.slns[index][:self._nstages]
        b = self.slns[index][self._nstages:]
        if n == 1:
            return TpMax(self._njobs, self._nstages, self._burnin, r, b, self.simulator)
        else:
            results = [TpMax(self._njobs, self._nstages, self._burnin, r, b, self.simulator) for i in range(n)]
            return np.array(results)
        
    def syscount(self):
        return len(self.slns)
    
    def getbest(self):
        return self.best_mean


def TpMax_pause(_njobs,  _nstages, _burnin, r, b, simulator):
    ptime.sleep(0.0005 + np.random.rand()*0.0015)
    sTime1 = np.random.exponential(1/r[0], _njobs)
    sTime2 = np.random.exponential(1/r[1], _njobs)
    sTime3 = np.random.exponential(1/r[2], _njobs)
    b1 = b[0]
    b2 = b[1]
    return simulator(_njobs, _burnin, sTime1, sTime2, sTime3, b1, b2)

    
class TpMaxGeneratorPause(object):
    def __init__(self, _njobs,  _nstages, _burnin, slns, means, simulator):
        self._njobs = _njobs
        self._nstages = _nstages
        self._burnin = _burnin
        # suffle the orders
        ids = np.arange(len(slns))
        self.slns = slns
        self.means = means
        self.best_mean = np.max(self.means)
        self.simulator = simulator
        
    def get(self, index, n=1):
        r = self.slns[index][:self._nstages]
        b = self.slns[index][self._nstages:]
        if n == 1:
            return TpMax_pause(self._njobs, self._nstages, self._burnin, r, b, self.simulator)
        else:
            results = [TpMax_pause(self._njobs, self._nstages, self._burnin, r, b, self.simulator) for i in range(n)]
            return np.array(results)
        
    def syscount(self):
        return len(self.slns)
    
    def getbest(self):
        return self.best_mean


def process_result(generator, num_cores, results, phasetimeset, simutimeset):

    seeding_times, exploration_times, greedy_times = phasetimeset
    simu_times1, simu_times2, simu_times3 = simutimeset

    seeding = np.mean(seeding_times)
    exploration = np.mean(exploration_times)
    greedy = np.mean(greedy_times)
    simu1 = np.mean(simu_times1)
    simu2 = np.mean(simu_times2)
    simu3 = np.mean(simu_times3)

    util1 = simu1 / seeding / num_cores
    util2 = simu2 / exploration / num_cores
    util3 = simu3 / greedy / num_cores
    print("##### Utilization ratios for the stages: {:.2%}, {:.2%}, {:.2%} #####".format(util1, util2, util3))
    util = (simu1 + simu2 + simu3) / (seeding +exploration + greedy) / num_cores
    print("##### Procedure utilization: {:.2%} #####".format(util))
    # utilset = [util1, util2, util3, util]
    PCS = evaluate_PCS_parallel(generator, results)
    PGS = evaluate_PGS_parallel(generator, results)
    print("******* PCS: {}, PGS: {} *******".format(PCS, PGS))
    # print(utilset)
    names = ["seeding time:", "exploration time:", "greedy time:", "simulation time in seeding:", 
                "simulation time in exploration:", "simulation time in greedy:",
                "total time:", "total simulation time:"]
    timeset = phasetimeset + simutimeset
    timeset.append(np.sum(phasetimeset, axis=0))
    timeset.append(np.sum(simutimeset, axis=0))
    for i, times in enumerate(timeset):
        average = np.mean(times)
        CI = st.norm.interval(alpha=0.95, loc=np.mean(times),scale=st.sem(times))
        print(names[i] + " {:.3f} ".format(average) + u"\u00B1" +" {:.3f}".format(average-CI[0]))
