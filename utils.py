
import numpy as np 
import time
import ray

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


def TpMax(_njobs,  _nstages, _burnin, r, b, simulator):
    sTime1 = np.random.exponential(1/r[0], _njobs)
    sTime2 = np.random.exponential(1/r[1], _njobs)
    sTime3 = np.random.exponential(1/r[2], _njobs)
    b1 = b[0]
    b2 = b[1]
    return simulator(_njobs, _burnin, sTime1, sTime2, sTime3, b1, b2)

class TpMaxGenerator(object):
    def __init__(self, _njobs,  _nstages, _burnin, slns, means, simulator):
        self._njobs = _njobs
        self._nstages = _nstages
        self._burnin = _burnin
        # suffle the orders
        ids = np.arange(len(slns))
        rids = np.random.permutation(ids)
        self.slns = slns[rids]
        self.means = means[rids]
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
    
#     def PCS(self, return_ids):
#         size = len(return_ids)
#         n_correct = 0
#         for _id in return_ids:
#             if self.means[_id]  == self.best_mean:
#                 n_correct += 1
#         return n_correct/size

#     def PGS(self, return_ids, delta=0.01):
#         size = len(return_ids)
#         n_good = 0
#         for _id in return_ids:
#             if  self.best_mean - self.means[_id] < delta:
#                 n_good += 1
#         return n_good/size
