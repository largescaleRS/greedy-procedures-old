from multiprocessing import Process, Queue
import numpy as np
import copy
from tqdm import tqdm

import sys

import sys
if sys.platform =='win32':
    import time as time
    import win_precise_time as ptime
elif sys.platform =='darwin':
    import time as time
    import time as ptime


def sequential_filling(allocations, num_cores):
    k = len(allocations)
    L = np.array(allocations)
    V = np.ones(num_cores) * np.sum(allocations)/num_cores
    V = np.ceil(V).astype(int)
    assignments = np.zeros((num_cores, k))
    j = 0
    for i in range(k):
        while True:
            if  L[i] <= V[j]:
                assignments[j, i] += L[i]
                V[j] -= L[i]
                break
            else:
                assignments[j, i] += V[j]
                L[i] -= V[j]
                j += 1
    return assignments.astype(int).tolist()


def __EFGPlusPlus(_id, _seed, batch_size, generator, private_q, common_q):
    np.random.seed(_seed)
    k = generator.syscount()
    while True:
        result = private_q.get(block=True)
        mode = result[0]
        if mode == 0:
            common_q.put(0)
        if mode == 1:
            # receive a task assignment
            simulation_time = 0
            assignments = result[1]
            sums = []
            for i in range(k):
                if assignments[i] > 0:
                    start_time = time.time()
                    sums.append(np.sum(generator.get(i, assignments[i])))
                    simulation_time += time.time() - start_time
                else:
                    sums.append(np.nan)
            common_q.put([sums, simulation_time])
        if mode == 2:
            selected = result[1]
            start_time = time.time()
            batch_sum = np.sum(generator.get(selected, batch_size))
            simulation_time = time.time() - start_time
            common_q.put([_id, selected, batch_sum, simulation_time])
        if mode == 3:
            common_q.put(3)
            break
    return


def EFGPlusPlus(generator, nsd, n0, ng, G, num_cores, batch_size=1, n_reps=1, seed=0, is_asyn=False):

    k = generator.syscount()
    rng = np.random.default_rng(seed)

    print("\nTotal number of alternatives - {}".format(k))
    print("Seeding/Exploration/Greedy budget - B={}k, B={}k, B={}k".format(nsd, n0, ng))
    print("Number of groups for seeding - {}".format(G))
    print("Number of processors used - {}".format(num_cores))
    print("Greedy batch size in each processor - {}".format(batch_size))
    print("Total replications to conduct - {}".format(n_reps))
    print("Asyn version? {}".format(is_asyn))

    # copy the generators
    generators = [copy.deepcopy(generator) for i in range(num_cores)]

    common_q = Queue(1000)
    private_qs = [Queue(1000) for i in range(num_cores)]

    start = time.time()
    print("Initializing {} processors...".format(num_cores))
    processors = [Process(target=__EFGPlusPlus, args=(i, rng.integers(1, 10e8), batch_size, 
                                generators[i], private_qs[i], common_q)) for i in range(num_cores)]
    # start the processors
    for p in processors:
        p.start()

    # initialize the processors
    for i in range(num_cores):
        private_qs[i].put([0])
    for i in range(num_cores):
        _ = common_q.get(block=True)

    end = time.time()
    print("Initializing the processors takes {:.3f}s".format(end-start))

    seeding_times = []
    exploration_times = []
    greedy_times = []

    estimated_bestids = []
    estimated_bestmeans = []

    simulation_times1 = []
    simulation_times2 = []
    simulation_times3 = []

    assign_duration1_times = []
    assign_duration2_times = []

    for rep in tqdm(range(n_reps)):
        # print("Running the {:.3f} replication......".format(rep+1))

        simulation_time1 = 0
        simulation_time2 = 0
        simulation_time3 = 0
        start_time = time.time()
        
        # seeding phase
        allocations = np.int64(np.ones(k)*nsd)
        assignments = sequential_filling(allocations, num_cores)
        
        for i in range(num_cores):
            private_qs[i].put([1, assignments[i]])

        end_time = time.time()
        assign_duration1 = end_time - start_time
        start_time = time.time()

        _sums = []
        for i in range(num_cores):
            result = common_q.get(block=True)
            _sums.append(result[0])
            simulation_time1 += result[1]
        
        sample_means = np.nansum(np.array(_sums), axis=0) / allocations

        end_time = time.time()
        seeding_duration = end_time - start_time
        start_time = time.time()

        # print("The seeding phase takes {:.3f}s".format(seeding_duration))

        # seeding information based allocation
        sorted_ids = np.argsort(-sample_means)
        group_sizes = [2**r for r in np.arange(G)]
        segments = group_sizes / np.sum(group_sizes) * k
        segments = np.int32(np.floor(segments))
        segments[-1] = k - np.sum(segments[:-1])
        allocations = n0*(2**G-1)/G/2**np.arange(G)
        allocations = np.floor(allocations)
        _regime = []
        for i in range(G):
            _regime.append(np.ones(segments[i])*allocations[i])
        sample_allocations = np.int32(np.concatenate(_regime))
        
        allocations = np.zeros(k)
        allocations[sorted_ids] = sample_allocations
        assignments = sequential_filling(allocations, num_cores)

        # exploration phase
        for i in range(num_cores):
            private_qs[i].put([1, assignments[i]])

        end_time = time.time()
        assign_duration2 = end_time - start_time
        start_time = time.time()

        _sums = []
        for i in range(num_cores):
            result = common_q.get(block=True)
            _sums.append(result[0])
            simulation_time2 += result[1]
        
        sample_means = np.nansum(_sums, axis=0) / allocations
        sample_sizes = allocations

        end_time = time.time()
        exploration_duration = end_time - start_time
        start_time = time.time()

        # print("The exploration phase takes {:.3f}s".format(exploration_duration))

        if is_asyn:
            # Asyn greedy phase
            N = (ng+n0)*k
            used = np.sum(sample_sizes) + num_cores*batch_size
            
            selected = np.argmax(sample_means)
            for i in range(num_cores):
                private_qs[i].put([2, selected])
            
            while used < N:
                result = common_q.get(block=True)
                _core_id = result[0]
                selected = result[1]
                _sum = result[2]
                simulation_time3 += result[3]
                sample_mean = sample_means[selected]
                sample_size = sample_sizes[selected]
                sample_mean = (sample_mean*sample_size + _sum)/(sample_size + batch_size)
                sample_size = sample_size + batch_size
                sample_means[selected]  = sample_mean
                sample_sizes[selected] = sample_size
                selected = np.argmax(sample_means)
                private_qs[_core_id].put((2, selected))
                used += batch_size
            
            for i in range(num_cores):
                result = common_q.get(block=True)
                _core_id = result[0]
                selected = result[1]
                _sum = result[2]
                simulation_time3 += result[3]
                # send out the new task
                sample_mean = sample_means[selected]
                sample_size = sample_sizes[selected]
                sample_mean = (sample_mean*sample_size + _sum)/(sample_size + batch_size)
                sample_size = sample_size + batch_size
                sample_means[selected]  = sample_mean
                sample_sizes[selected] = sample_size

            # select the final best
            estimated_bestid = np.argmax(sample_means)
            end_time = time.time()
            greedy_duration = end_time - start_time
            # print("The Asyn greedy phase takes {:.3f}s".format(greedy_duration))

        else:
            # greedy phase
            N = (ng+n0)*k
            used = np.sum(sample_sizes)
            while used < N:
                selected = np.argmax(sample_means)
                for i in range(num_cores):
                    private_qs[i].put([2, selected])
                _sum = 0
                for i in range(num_cores):
                    result = common_q.get(block=True)
                    _sum += result[2]
                    simulation_time3 += result[3]
                sample_mean = sample_means[selected]
                sample_size = sample_sizes[selected]
                sample_mean = (sample_mean*sample_size + _sum)/(sample_size + batch_size*num_cores)
                sample_size = sample_size + batch_size*num_cores
                sample_means[selected]  = sample_mean
                sample_sizes[selected] = sample_size
                used += batch_size*num_cores

            # select the final best
            estimated_bestid = np.argmax(sample_means)
            end_time = time.time()
            greedy_duration = end_time - start_time
            # print("The greedy phase takes {:.3f}s".format(greedy_duration))

        # select the final best mean and log the times
        estimated_bestids.append(estimated_bestid)
        estimated_bestmeans.append(sample_means[estimated_bestid])

        seeding_times.append(seeding_duration)
        exploration_times.append(exploration_duration)
        greedy_times.append(greedy_duration)

        simulation_times1.append(simulation_time1)
        simulation_times2.append(simulation_time2)
        simulation_times3.append(simulation_time3)
        assign_duration1_times.append(assign_duration1)
        assign_duration2_times.append(assign_duration2)

    phasetimeset = [seeding_times, exploration_times, greedy_times]
    simutimeset = [simulation_times1, simulation_times2, simulation_times3]
    
    # close the processors
    for i in range(num_cores):
        private_qs[i].put([3])
    for i in range(num_cores):
        common_q.get(block=True)

    for p in processors:
        p.join()
    for p in processors:
        p.close()

    return estimated_bestids, phasetimeset, simutimeset