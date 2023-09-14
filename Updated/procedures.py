import ray
import numpy as np



def EFGPlusPlus(generator, nsd, n0, ng, G=0, M=0, seed=0, expe_id=-1, **args):

    k = generator.syscount()
    ids = np.arange(k)
    
    if M==0:
        M = int(np.sqrt(k)/3)
        
    if G==0:
        G = maxG(k)
        
    np.random.seed(seed)

    counts = np.zeros(k)
    sample_means =  np.zeros(k)
    
    # initial stage seeding 
    for _id in ids:
        sample_means[_id] = np.mean(generator.get(_id, nsd))
        
    sorted_ids = np.argsort(-sample_means)
    total_count = nsd*k
    # sampling budget allocation according to seeding rank
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
    total_count += np.sum(sample_allocations)
    for i, _id in enumerate(sorted_ids):
        sample_means[_id] = np.mean(generator.get(_id, sample_allocations[i]))
        counts[_id] = sample_allocations[i]
        
    # randomly allocate alternatives
    ids = np.arange(k)
    np.random.shuffle(ids)
    assign_cores_ids = np.array_split(ids, M)

    # in core process
    delivered_ids = []
    delivered_means = []
    for core in range(M):
        core_ids = assign_cores_ids[core]
        core_k = len(core_ids)
        core_sample_means = sample_means[core_ids].copy()
        core_counts = counts[core_ids].copy()
        N = ng*core_k
        used = 0
        while used < N:
            selected = np.argmax(core_sample_means)
            sample = generator.get(core_ids[selected])
            core_sample_means[selected] = core_counts[selected] * core_sample_means[selected] + sample
            core_counts[selected] += 1
            core_sample_means[selected]  = core_sample_means[selected] / core_counts[selected]
            used += 1
            total_count += 1
        estimated_best = np.nanargmax(core_sample_means)
        delivered_ids.append(core_ids[estimated_best])
        delivered_means.append(core_sample_means[estimated_best])
        

    estimated_bestid = delivered_ids[np.nanargmax(delivered_means)]
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid
    
@ray.remote    
def remote_EFGPlusPlus(generator, nsd, n0, ng, G=0, M=0, seed=0, expe_id=-1, **args):

    k = generator.syscount()
    ids = np.arange(k)
    
    if M==0:
        M = int(np.sqrt(k)/3)
    if G==0:
        G = maxG(k)
        
    np.random.seed(seed)

    counts = np.zeros(k)
    sample_means =  np.zeros(k)
    
    # initial stage seeding 
    for _id in ids:
        sample_means[_id] = np.mean(generator.get(_id, nsd))
        
    sorted_ids = np.argsort(-sample_means)
    total_count = nsd*k
    # sampling budget allocation according to seeding rank
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
    total_count += np.sum(sample_allocations)
    for i, _id in enumerate(sorted_ids):
        sample_means[_id] = np.mean(generator.get(_id, sample_allocations[i]))
        counts[_id] = sample_allocations[i]
        
    # randomly allocate alternatives
    ids = np.arange(k)
    np.random.shuffle(ids)
    assign_cores_ids = np.array_split(ids, M)
    
    # in core process
    delivered_ids = []
    delivered_means = []
    for core in range(M):
        core_ids = assign_cores_ids[core]
        core_k = len(core_ids)
        core_sample_means = sample_means[core_ids].copy()
        core_counts = counts[core_ids].copy()
        N = ng*core_k
        used = 0
        while used < N:
            selected = np.argmax(core_sample_means)
            sample = generator.get(core_ids[selected])
            core_sample_means[selected] = core_counts[selected] * core_sample_means[selected] + sample
            core_counts[selected] += 1
            core_sample_means[selected]  = core_sample_means[selected] / core_counts[selected]
            used += 1
            total_count += 1
        estimated_best = np.nanargmax(core_sample_means)
        delivered_ids.append(core_ids[estimated_best])
        delivered_means.append(core_sample_means[estimated_best])
        

    estimated_bestid = delivered_ids[np.nanargmax(delivered_means)]
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid

def SH(generator, n, expe_id=-1, seed=0):
    k = generator.syscount()
    np.random.seed(seed)
    # check condition n / factor**(factor-1) >= 1
    
    R = np.int32(np.ceil(np.log2(k)))
    Ir = np.arange(k)
#     counts = np.zeros(k)
#     sample_means = np.zeros(k)
#     n_samples = 0

    for round in range(1, R+1):
        stage_size = np.int32(np.floor(n*k/(R*len(Ir))))
        stage_means = []
        for _id in Ir:
            temp_mean = np.mean(generator.get(_id, stage_size))
            stage_means.append(temp_mean)
#             sample_means[_id] = temp_mean
#             counts[_id] = stage_size
        stage_means = np.array(stage_means)
        _sorted = np.argsort(-stage_means)
        n_left = np.int32(np.ceil(len(Ir)/2))
        # how many used
#         n_samples += stage_size * len(Ir)
        # enter into next round
#         print(n_left, _sorted)
        Ir = Ir[_sorted[:n_left]]
    
    estimated_bestid = Ir[0]
    
    print(Ir)
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid

@ray.remote
def remote_SH(generator, n, expe_id=-1, seed=0):
    k = generator.syscount()
    np.random.seed(seed)
    
    R = np.int32(np.ceil(np.log2(k)))
    Ir = np.arange(k)

    for round in range(1, R+1):
        stage_size = np.int32(np.floor(n*k/(R*len(Ir))))
        stage_means = []
        for _id in Ir:
            temp_mean = np.mean(generator.get(_id, stage_size))
            stage_means.append(temp_mean)
        stage_means = np.array(stage_means)
        _sorted = np.argsort(-stage_means)
        n_left = np.int32(np.ceil(len(Ir)/2))
        Ir = Ir[_sorted[:n_left]]
    
    estimated_bestid = Ir[0]
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid



def ModifiedSH(generator, n, expe_id=-1, seed=0):
    k = generator.syscount()
    np.random.seed(seed)

    L = np.int32(np.ceil(np.log2(k)))
    Ir = np.arange(k)
#     counts = np.zeros(k)
#     sample_means = np.zeros(k)
    n_samples = 0

    for round in range(1, L+1):
#         stage_size = np.int32(np.floor(n*k/(R*len(Ir))))
        stage_size = np.int32(np.floor(n/81*(16/9)**(round-1)*round))
        stage_means = []
        for _id in Ir:
            temp_mean = np.mean(generator.get(_id, stage_size))
            stage_means.append(temp_mean)
#             sample_means[_id] = temp_mean
#             counts[_id] = stage_size
        stage_means = np.array(stage_means)
        _sorted = np.argsort(-stage_means)
        n_left = np.int32(np.ceil(len(Ir)/2))
        # how many used
        n_samples += stage_size * len(Ir)
        # enter into next round
#         print(n_left, _sorted)
        Ir = Ir[_sorted[:n_left]]
    
    estimated_bestid = Ir[0]
    
    print(Ir, n*k, n_samples)
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid
    
    
@ray.remote
def remote_ModifiedSH(generator, n, expe_id=-1, seed=0):
    k = generator.syscount()
    np.random.seed(seed)
    # check condition n / factor**(factor-1) >= 1
    
    L = np.int32(np.ceil(np.log2(k)))
    Ir = np.arange(k)
#     counts = np.zeros(k)
#     sample_means = np.zeros(k)
    n_samples = 0

    for round in range(1, L+1):
#         stage_size = np.int32(np.floor(n*k/(R*len(Ir))))
        stage_size = np.int32(np.floor(n/81*(16/9)**(round-1)*round))
        stage_means = []
        for _id in Ir:
            temp_mean = np.mean(generator.get(_id, stage_size))
            stage_means.append(temp_mean)
#             sample_means[_id] = temp_mean
#             counts[_id] = stage_size
        stage_means = np.array(stage_means)
        _sorted = np.argsort(-stage_means)
        n_left = np.int32(np.ceil(len(Ir)/2))
        # how many used
        n_samples += stage_size * len(Ir)
        # enter into next round
#         print(n_left, _sorted)
        Ir = Ir[_sorted[:n_left]]
    
    estimated_bestid = Ir[0]
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid
    

def EFGPlus(generator, nsd, n0, ng, G, seed=0, expe_id=-1, **args):

    k = generator.syscount()
    ids = np.arange(k)
    
    np.random.seed(seed)

    counts = np.zeros(k)
    sample_means =  np.zeros(k)
    
    # initial stage seeding 
    for _id in ids:
        sample_means[_id] = np.mean(generator.get(_id, nsd))
    sorted_ids = np.argsort(-sample_means)
    # sampling budget allocation according to seeding rank
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
    print(np.sum(sample_allocations))
    for i, _id in enumerate(sorted_ids):
        sample_means[_id] = np.mean(generator.get(_id, sample_allocations[i]))
        counts[_id] = sample_allocations[i]
    
    N = (n0+ng)*k
    used = np.sum(sample_allocations)
    
    while used < N:
        selected = np.argmax(sample_means)
        sample = generator.get(selected)
        sample_means[selected] = counts[selected] * sample_means[selected] + sample
        counts[selected] += 1
        sample_means[selected]  = sample_means[selected] / counts[selected]
        used += 1
        
    estimated_bestid = np.nanargmax(sample_means)
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid
    


@ray.remote
def remote_EFGPlus(generator, nsd, n0, ng, G, seed=0, expe_id=-1, **args):

    k = generator.syscount()
    ids = np.arange(k)
    
    np.random.seed(seed)

    counts = np.zeros(k)
    sample_means =  np.zeros(k)
    
    # initial stage seeding 
    for _id in ids:
        sample_means[_id] = np.mean(generator.get(_id, nsd))
    sorted_ids = np.argsort(-sample_means)
    # sampling budget allocation according to seeding rank
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
    for i, _id in enumerate(sorted_ids):
        sample_means[_id] = np.mean(generator.get(_id, sample_allocations[i]))
        counts[_id] = sample_allocations[i]
    
    N = (n0+ng)*k
    used = np.sum(sample_allocations)
    
    while used < N:
        selected = np.argmax(sample_means)
        sample = generator.get(selected)
        sample_means[selected] = counts[selected] * sample_means[selected] + sample
        counts[selected] += 1
        sample_means[selected]  = sample_means[selected] / counts[selected]
        used += 1
        
    estimated_bestid = np.nanargmax(sample_means)
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid



def FBKT(generator, nsd=0, n=10, phi=3, seeding=False, seed=0, expe_id=-1, **args):
    # nsd K for seeding
    K = generator.syscount()
    np.random.seed(seed)
    # initial set of alternatives (id)
    Ir = np.arange(K)
    # number of total rounds
    l = int(np.log2(K)) + 1
    print("total rounds: {}".format(l))
    # cumulative means and counts
    c_means = np.zeros(K)
    c_counts = np.zeros(K)
    if seeding:
        for _id in Ir:
            c_means[_id] = np.mean(generator.get(_id, nsd))
        c_counts = np.ones(K)*nsd
    # total budget for non-seeding operations
    N = n*K
    used = 0
    # begin the stage-wise competition
    for r in range(1, l+1):
        Nr = int(r/phi/(phi-1)*((phi-1)/phi)**r *N)
        stage_size = int(Nr/len(Ir))
        print('budget allocation for round {}: {}'.format(r, stage_size))
        # if not seeding
        if seeding:
            # ranking the alternatives by the means before this round
            c_means_r = c_means[Ir]
            a_order = np.argsort(c_means_r)
            _Ir = Ir[a_order]
        else:
            _Ir = np.random.permutation(Ir)
        Ir_next = []
        n_competitors = len(_Ir)
        if n_competitors % 2 == 0:
            # even number of competitors
            pass
        else:
            # the last one is get into next round
            Ir_next.append(_Ir[-1])
        n_matchs  = np.int64(n_competitors/2)
        # how many sampling budget used here
        used += n_matchs*2*stage_size
        competitorsA = _Ir[:n_matchs]
        competitorsB = _Ir[n_matchs:2*n_matchs]
        if seeding:
            competitorsB = competitorsB[::-1]
        for m in range(n_matchs):
            A = competitorsA[m]
            B = competitorsB[m]
            A_mean = np.mean(generator.get(A, stage_size))
            B_mean = np.mean(generator.get(B, stage_size))
            if A_mean >= B_mean:
                Ir_next.append(A)
                if seeding:
                    # update the c_means, counts
                    c_means[A] = (c_means[A] * c_counts[A] + A_mean * stage_size) / (stage_size + c_counts[A])
                    c_counts[A] = c_counts[A] + stage_size
            else:
                Ir_next.append(B)
                if seeding:
                    c_means[B] = (c_means[B] * c_counts[B] + B_mean * stage_size) / (stage_size + c_counts[B])
                    c_counts[B] = c_counts[B] + stage_size
        # alternatives for the next round
        Ir = np.array(Ir_next).astype(int)
        if len(Ir) == 1:
            break

    print("Problem Size:{}, total samples taken: {}, effective n_0:{}".format(K, used, used/K))
    estimated_bestid = Ir[0]
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid
        
@ray.remote
def remote_FBKT(generator, nsd=10, n=0, phi=3, seeding=False, seed=0, expe_id=-1, **args):
    # nsd K for seeding
    K = generator.syscount()
    np.random.seed(seed)
    # initial set of alternatives (id)
    Ir = np.arange(K)
    # number of total rounds
    l = int(np.log2(K)) + 1
    # cumulative means and counts
    c_means = np.zeros(K)
    c_counts = np.zeros(K)
    if seeding:
        for _id in Ir:
            c_means[_id] = np.mean(generator.get(_id, nsd))
        c_counts = np.ones(K)*nsd
    # total budget for non-seeding operations
    N = n*K
    # begin the stage-wise competition
    for r in range(1, l+1):
        Nr = int(r/phi/(phi-1)*((phi-1)/phi)**r *N)
        stage_size = int(Nr/len(Ir))
        # if not seeding
        if seeding:
            # ranking the alternatives by the means before this round
            c_means_r = c_means[Ir]
            a_order = np.argsort(c_means_r)
            _Ir = Ir[a_order]
        else:
            _Ir = np.random.permutation(Ir)
        Ir_next = []
        n_competitors = len(_Ir)
        if n_competitors % 2 == 0:
            # even number of competitors
            pass
        else:
            # the last one is get into next round
            Ir_next.append(_Ir[-1])
        n_matchs  = np.int64(n_competitors/2)
        competitorsA = _Ir[:n_matchs]
        competitorsB = _Ir[n_matchs:2*n_matchs]
        if seeding:
            competitorsB = competitorsB[::-1]
        for m in range(n_matchs):
            A = competitorsA[m]
            B = competitorsB[m]
            A_mean = np.mean(generator.get(A, stage_size))
            B_mean = np.mean(generator.get(B, stage_size))
            if A_mean >= B_mean:
                Ir_next.append(A)
                if seeding:
                    # update the c_means, counts
                    c_means[A] = (c_means[A] * c_counts[A] + A_mean * stage_size) / (stage_size + c_counts[A])
                    c_counts[A] = c_counts[A] + stage_size
            else:
                Ir_next.append(B)
                if seeding:
                    c_means[B] = (c_means[B] * c_counts[B] + B_mean * stage_size) / (stage_size + c_counts[B])
                    c_counts[B] = c_counts[B] + stage_size
        # alternatives for the next round
        Ir = np.array(Ir_next).astype(int)
        if len(Ir) == 1:
            break
            
    estimated_bestid = Ir[0]
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid
        
        
        
def EFG(generator, n0, ng, seed=0, expe_id=-1, **args):

    K = generator.syscount()
    ids = np.arange(K)
    
    np.random.seed(seed)

    counts = np.zeros(K)
    sample_means =  np.zeros(K)

    N = (n0+ng)*K

    used = n0*K
    
    for _id in ids:
        sample_means[_id] = np.mean(generator.get(_id, n0))
    counts = np.ones(K) * n0
    
    while used < N:
        selected = np.argmax(sample_means)
        sample = generator.get(selected)
        sample_means[selected] = counts[selected] * sample_means[selected] + sample
        counts[selected] += 1
        sample_means[selected]  = sample_means[selected] / counts[selected]
        used += 1
        
    estimated_bestid = np.nanargmax(sample_means)
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid
    
@ray.remote
def remote_EFG(generator, n0, ng, seed=0, expe_id=-1, **args):

    K = generator.syscount()
    ids = np.arange(K)
    
    np.random.seed(seed)

    counts = np.zeros(K)
    sample_means =  np.zeros(K)

    N = (n0+ng)*K

    used = n0*K
    
    for _id in ids:
        sample_means[_id] = np.mean(generator.get(_id, n0))
    counts = np.ones(K) * n0
    
    while used < N:
        selected = np.argmax(sample_means)
        sample = generator.get(selected)
        sample_means[selected] = counts[selected] * sample_means[selected] + sample
        counts[selected] += 1
        sample_means[selected]  = sample_means[selected] / counts[selected]
        used += 1
        
    estimated_bestid = np.nanargmax(sample_means)
    
    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid


def OCBA(generator, n, ng=0, seed=0, expe_id=-1,**args):
    
    # can not apply to the problem
    K = generator.syscount()
    np.random.seed(seed)
    ids = np.arange(K)

    counts = np.zeros(K)
    sample_means =  np.zeros(K)
    
    N = n * K

    means = generator.means
    variances = generator.variances
    best = np.argmax(means)

    mean_differences = means[best] - means
    mean_differences[best] = 1 # aviod zero
    coefs = variances / (mean_differences) ** 2
    mean_differences[best] = 0 # recover zero
    coefs[best] = 10e9
    nonbestminimal = np.argmin(coefs) # find the min coef one
    times = coefs / coefs[nonbestminimal] # 
    times[best] = 0
    times[best] = np.sqrt(np.sum(variances[best]/variances * times ** 2))
    allocations = N / np.sum(times) * times
    allocations = np.ceil(allocations).astype(int) 
    # allocations[best] += N-np.sum(allocations)
    for _id in ids:
        if allocations[_id] > 0:
            sample_means[_id] = np.mean(generator.get(_id, allocations[_id]))
        else:
            sample_means[_id] = -10e8
    counts = allocations.copy()

    # greedy part
    N = int(ng * K)
    used = 0
    while used < N:
        selected = np.argmax(sample_means)
        sample = generator.get(selected)
        sample_means[selected] = counts[selected] * sample_means[selected] + sample
        counts[selected] += 1
        sample_means[selected]  = sample_means[selected] / counts[selected]
        used += 1

    print("Problem Size:{}, total samples taken: {}, effective c:{}".format(K, np.sum(counts), np.sum(counts)/K))

    estimated_bestid = np.nanargmax(sample_means)

    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid


@ray.remote
def remote_OCBA(generator, n, ng=0, seed=0, expe_id=-1,**args):
    
    # can not apply to the problem
    K = generator.syscount()
    np.random.seed(seed)
    ids = np.arange(K)

    counts = np.zeros(K)
    sample_means =  np.zeros(K)
    
    N = n * K

    means = generator.means
    variances = generator.variances
    best = np.argmax(means)

    mean_differences = means[best] - means
    mean_differences[best] = 1 # aviod zero
    coefs = variances / (mean_differences) ** 2
    mean_differences[best] = 0 # recover zero
    coefs[best] = 10e9
    nonbestminimal = np.argmin(coefs) # find the min coef one
    times = coefs / coefs[nonbestminimal] # 
    times[best] = 0
    times[best] = np.sqrt(np.sum(variances[best]/variances * times ** 2))
    allocations = N / np.sum(times) * times
    allocations = np.ceil(allocations).astype(int) 
    # allocations[best] += N-np.sum(allocations)
    for _id in ids:
        if allocations[_id] > 0:
            sample_means[_id] = np.mean(generator.get(_id, allocations[_id]))
        else:
            sample_means[_id] = -10e8
    counts = allocations.copy()

    # greedy part
    N = int(ng * K)
    used = 0
    while used < N:
        selected = np.argmax(sample_means)
        sample = generator.get(selected)
        sample_means[selected] = counts[selected] * sample_means[selected] + sample
        counts[selected] += 1
        sample_means[selected]  = sample_means[selected] / counts[selected]
        used += 1
        
    estimated_bestid = np.nanargmax(sample_means)

    if expe_id >= 0:
        return expe_id, estimated_bestid
    else:
        return estimated_bestid