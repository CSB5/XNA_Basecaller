#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

BASE_MAP = ['N','A','C','G','T','X','Y']

BASE_DIR = os.path.expanduser('.')
DEFAULT_MODEL = os.path.join(BASE_DIR, 'ub-bonito/bonito/data',
                             'r9.4_450bps.nucleotide.6mer.XNA-Px_Ds.template.model')
def load_kmer_model(ref_filepath=DEFAULT_MODEL):
    if ref_filepath is None:
        ref_filepath = DEFAULT_MODEL
    model_kmer_df = pd.read_csv(ref_filepath, sep='\t', comment='#')
    tight_dict = model_kmer_df.set_index('kmer')[['level_mean','level_stdv']].to_dict('split')
    kmer_poremodel = { k:v for k,v in zip(tight_dict['index'], tight_dict['data']) }
    return kmer_poremodel

# Old sequence_official_poremodel
def get_kmers_model(sequence, kmer_poremodel, k, append=True):
    ### Append tail with ATs (most common and not homopolymer)
    if append:
        if sequence[-1] != 'A':
            sequence += 'ATATA'
        else:
            sequence += 'TATAT'
        
    length=len(sequence)
    if length < k:
        # Assign mean and std value
        kmer_means=list()
        kmer_stdvs=list()
        [kmer_means.extend((float(90.2083),)*1) for i in range(length)]
        [kmer_stdvs.extend((float(2.0),)*1) for i in range(length)]
    else:
        kmers = [sequence[i:i + k] for i in range(0, length - k + 1)]
        kmer_means, kmer_stdvs = zip(*[kmer_poremodel[kmer] for kmer in kmers])
        kmer_means = list(kmer_means)
        kmer_stdvs = list(kmer_stdvs)

    return kmer_means, kmer_stdvs

def compute_med_mad_squiggly(means, stds, kmer_rep=100, factor=1.4826, rng=np.random):
    rep_stds = np.repeat(stds, kmer_rep)
    event_stds = rng.uniform(-1*rep_stds, rep_stds)
    # event_stds = rng.normal(0, rep_stds)
    # event_stds = np.clip(event_stds, -2*rep_stds, 2*rep_stds)
    squiggly_signal = np.repeat(means, kmer_rep) + event_stds
    med = np.median(squiggly_signal)
    mad = np.median(np.absolute(squiggly_signal - med)) * factor + np.finfo(np.float32).eps
    return med, mad

def sim_signals(seq, kmer_reps, kmer_poremodel, std_dist='uniform', 
                noise_std=0, variable_noise=False, append=False,
                kmer_len=6, rng=np.random):
    ### a) UB was spiked before and signal already generated (affects med/mad from original seq)
    # subseq_st, subseq_en = (pos-(kmer_len-1), pos+1)
    # subseq_means, subseq_stds = (means[subseq_st:subseq_en], stds[subseq_st:subseq_en])
    ### b) Spike UB and generate signal now (preserves med/mad)
    
    subseq_means, subseq_stds = get_kmers_model(
        seq, kmer_poremodel, kmer_len, append=append)
    
    #### Stds addition (uniform or normal sampling)
    rep_stds = np.repeat(subseq_stds, kmer_reps)
    if std_dist == 'uniform':
        event_stds = rng.uniform(-1*rep_stds, rep_stds)
        # mult_std = 1; event_stds = rng.uniform(-1*mult_std*rep_stds, mult_std*rep_stds)
    elif std_dist == 'uniform_shift_not_shared': 
        ### Shift the stds by a random multiplier, to change the "center" of the sample window
        ### Each kmer will have a different shift value
        shift_range = 1.5 
        shift = rng.choice(np.arange(-shift_range, shift_range+.01, 0.5), 
                           size=len(subseq_stds))
        shift = np.repeat(shift, kmer_reps)
        event_stds = rng.uniform((shift-1)*rep_stds, (shift+1)*rep_stds)
    elif std_dist == 'uniform_shift_shared': 
        ### Shift the stds by a random multiplier, to change the "center" of the sample window
        ### Each all kmers from same spike pos will have the same shift value
        shift_range = 1.5
        shift = rng.choice(np.arange(-shift_range, shift_range+.01, 0.5))
        event_stds = rng.uniform((shift-1)*rep_stds, (shift+1)*rep_stds)
    elif std_dist.startswith('uniform_shift'):
        _, _, std_len, shift_range = std_dist.split('_')
        std_len = float(std_len)
        shift_range = float(shift_range)
        shift = rng.choice(np.arange(-shift_range, shift_range+.01, 0.5))
        event_stds = rng.uniform((-std_len+shift)*rep_stds, (std_len+shift)*rep_stds)
    elif std_dist == 'normal': # dist not truc, samples are clipped afterwards
        ### TO DO Change stds to actual normal dists?! R: Looks too noisy!
        mult_std = 0.5 # Halving std, to make it less squiggly 
        event_stds = rng.normal(0, mult_std*rep_stds)
        event_stds = np.clip(event_stds, -2*rep_stds, 2*rep_stds)
    elif std_dist.startswith('truncnorm_shift'):
        _, _, std_len, shift_range = std_dist.split('_')
        std_len = float(std_len)
        shift_range = float(shift_range)
        shift = rng.choice(np.arange(-shift_range, shift_range+.01, 0.5))
        event_stds = truncnorm.rvs(-std_len+shift, std_len+shift, 
                                   scale=rep_stds, random_state=rng)
    elif std_dist == 'truncnorm':
        ### change to scipy.stats.truncnorm (no need clip, better dist)
        std_trunc = 2
        event_stds = truncnorm.rvs(-std_trunc, std_trunc, scale=rep_stds, random_state=rng)
    elif std_dist == 'truncnorm_prerep': # Compute stdev before std repetition
        std_trunc = 2
        pick_subseq_stds = truncnorm.rvs(-std_trunc, std_trunc, scale=subseq_stds, random_state=rng) 
        # print(subseq_stds[0], pick_subseq_stds[0])
        event_stds = np.repeat(pick_subseq_stds, kmer_reps)
    squiggly_signals = np.repeat(subseq_means, kmer_reps) + event_stds
    
    #### Add gauss noise
    if noise_std > 0:
        if not variable_noise:
            ### V1
            # noise = rng.normal(0, noise_std, len(squiggly_signals))
            # noise = np.clip(noise, -2*noise_std, 2*noise_std)
            ### V2
            std_trunc = 3
            noise = truncnorm.rvs(-std_trunc, std_trunc, scale=noise_std, 
                                  size=len(squiggly_signals), random_state=rng)
        else:
            pos_noise_std = rng.uniform(0, noise_std)
            # print(pos_noise_std)
            ### V1
            # noise = rng.normal(0, pos_noise_std, len(squiggly_signals))
            # noise = np.clip(noise, -2*pos_noise_std, 2*pos_noise_std)
            ### V2
            std_trunc = 3
            noise = truncnorm.rvs(-std_trunc, std_trunc, scale=pos_noise_std, 
                                  size=len(squiggly_signals), random_state=rng)
        squiggly_signals += noise
    return squiggly_signals

def spike_chunk(chunk, length, target, breakpts, spiked_pos_ubs, kmer_poremodel, 
                noise_std=0, kmer_len=6, equal_kmer_reps=True, std_dist='uniform',
                variable_noise=False, rng=np.random):
    ### TO DO replace spiked bases here? Yes if using these means+stds? R: No
    ## med/mad would change, but possibly not much?
    # base_map_rev = dict(zip(BASE_MAP,range(len(BASE_MAP))))
    # target[list(spiked_pos.keys())] = [ base_map_rev[ub] for ub in spiked_pos_ubs.values() ]
    target_dec = [ BASE_MAP[i] for i in target ]
    target_str = ''.join(target_dec)
    ###  Pseudo-random seed (Moved to spike_read)
    # seed = int(sha256(target_str.encode('utf-8')).hexdigest(), 16) % 10**9
    # seed = length # Constant for same read, but will change for reads w/ diff lens
    # np.random.seed(seed) # Constant for same read, but will change for reads w/ diff tars
    
    means, stds = get_kmers_model(target_str, kmer_poremodel, kmer_len)
    med, mad = compute_med_mad_squiggly(means, stds, rng=rng)

    spiked_chunk = np.array(chunk)
    
    # for pos in spiked_pos:
    for pos, ub in spiked_pos_ubs.items():
        # 1) Extract kmers to be synthesized target[pos-5:pos]
        ## subseq_target = [ BASE_MAP[i] for i in target[pos-(kmer_len-1):pos+kmer_len] ]
        subseq_target = target_dec[pos-(kmer_len-1):pos+kmer_len]
    
        # 2) Extract slice position and len (how many signals are available) using bkps[pos-5:pos]
        chunk_st = breakpts[pos-kmer_len] if pos>=kmer_len else 0
        chunk_en = breakpts[pos]
        chunk_len = chunk_en - chunk_st
        # print(f"Chunk slice: {chunk_st}-{chunk_en} (len={chunk_len})")
    
        # 3) Synthesize distributing evenly(?) for this signal len
        ### Equal(ish) signal repetition per kmer
        if equal_kmer_reps:
            kmer_reps = np.full(kmer_len, chunk_len//kmer_len)
            kmer_reps[:chunk_len%kmer_len] += 1
        else:
            ### For viz is better to use same kmer_reps as dtw_bkps
            subseq_breakpts = breakpts[pos-kmer_len+1:pos+1] - breakpts[pos-kmer_len]
            kmer_reps = np.asarray([subseq_breakpts[0]] + np.diff(subseq_breakpts).tolist())
    
        #### Replace spiked bases here? Need to regenerate means/stds
        ### a) UB was spiked before and signal already generated (affects med/mad from original seq)
        # subseq_st, subseq_en = (pos-(kmer_len-1), pos+1)
        # subseq_means, subseq_stds = (means[subseq_st:subseq_en], stds[subseq_st:subseq_en])
        ### b) Spike UB and generate signal now (preserves med/mad)
        if ub != 'N': subseq_target[kmer_len-1] = ub
        squiggly_signals = sim_signals(''.join(subseq_target), kmer_reps, kmer_poremodel, 
            std_dist=std_dist, noise_std=noise_std, variable_noise=variable_noise,
            kmer_len=kmer_len, rng=rng)
        
        norm_signals = (squiggly_signals - med) / mad
        
        ### 4) replace signals at chunk[slice]
        spiked_chunk[chunk_st:chunk_en] = norm_signals
    
    return spiked_chunk

def choose_positions(length, n_pos, pad=5, rng=np.random, ubs_pos=None):
    valid_pos_mask = np.full(length, True)
    # Avoiding first+last 10 bases, as DTW segmentation at edges is messier
    valid_pos_mask[:10], valid_pos_mask[-10:] = False, False
    
    #### Avoid UBs
    if ubs_pos is not None and len(ubs_pos) > 0:
        for pos in ubs_pos:
            st, en = max(0, pos-2*pad), pos+2*pad+1 # Mult pad to give more space
            valid_pos_mask[st:en] = False
    
    spiked_pos = []
    for i in range(n_pos):
        valid_pos = np.where(valid_pos_mask)[0]
        if len(valid_pos) == 0: # no position available anymore
            break
        pos = rng.choice(valid_pos, 1)[0]
        st, en = max(0, pos-pad), pos+pad+1
        valid_pos_mask[st:en] = False
        spiked_pos.append(pos)
    spiked_pos.sort()
    return spiked_pos

def sim_target(target, breakpts, kmer_poremodel, equal_kmer_reps=True, 
               std_dist='uniform', noise_std=0, variable_noise=False, 
               kmer_len=6, rng=np.random):
    target_dec = [ BASE_MAP[i] for i in target ]
    target_str = ''.join(target_dec)
    
    means, stds = get_kmers_model(target_str, kmer_poremodel, kmer_len)
    med, mad = compute_med_mad_squiggly(means, stds, rng=rng)

    # sim_chunk = np.empty(breakpts[-1])
    if equal_kmer_reps:
        chunk_len = breakpts[-1]
        kmer_reps = np.full(kmer_len, chunk_len//kmer_len)
        kmer_reps[:chunk_len%kmer_len] += 1
    else:
        ### For viz is better to use same kmer_reps as dtw_bkps (better results also)
        kmer_reps = np.asarray([breakpts[0]] + np.diff(breakpts).tolist())

    #### Simulate spiked kmers means/stds
    squiggly_signals = sim_signals(target_str, kmer_reps, kmer_poremodel, 
        std_dist=std_dist, noise_std=noise_std, variable_noise=variable_noise,
        append=True, kmer_len=kmer_len, rng=rng)
    
    norm_signals = (squiggly_signals - med) / mad

    ### Default was float64, not suitable for Bonito train
    sim_chunk = np.asarray(norm_signals, dtype=np.float32)
    
    return sim_chunk

def spike_read(chunk, length, target, breakpts, prop_ubs, ubs, kmer_poremodel, 
               var_prop_ubs=None, fully_synth=False, rng=np.random, legacy_pos=False, 
               pad=5, mix_ubs=True, **syn_signal_kwargs):
               # noise_std=0, std_dist='uniform'):
    #### [OPT] chunk dtype to float16
    # chunk = np.asarray(chunk, dtype=np.float16)
    
    base_map_rev = dict(zip(BASE_MAP,range(len(BASE_MAP))))

    if var_prop_ubs is not None and var_prop_ubs > 0:
        prop_ubs = rng.uniform(prop_ubs-var_prop_ubs, prop_ubs+var_prop_ubs)
    
    #### Get UB positions to avoid spiking over them
    ubs_pos = np.argwhere(target[:length]>4)[:,0]

    # 2.1) choose spiked pos and UBs
    if legacy_pos: # Legacy number of ubs, to keep prop. same as initial exps
        n_pos = len(np.arange(10, length-10, step=(length-20)//round(length*prop_ubs)))
    else: # More accurate number of ubs to achieve desired prop_ubs
        n_pos = round(length*prop_ubs)
        # n_pos = np.ceil(length*prop_ubs).astype(int) # Ceil forces to add at least 1 UB
    n_pos -= len(ubs_pos) # Take into account quantity of UBs already present
    n_pos = max(n_pos, 1) # Force to have at least one UB added
    spiked_pos = choose_positions(length, n_pos, rng=rng, ubs_pos=ubs_pos, pad=pad)
    n_pos = len(spiked_pos) # Actual n_pos might change due to limited space for inserting UBs
    
    if mix_ubs:
        spiked_ubs = ubs*((n_pos+n_pos%2)//len(ubs))
        if len(ubs) > 1:
            rng.shuffle(spiked_ubs)
        spiked_ubs = spiked_ubs[:n_pos] # trim extra base (case when n_pos is odd)
    else:
        spiked_ubs = n_pos * [rng.choice(ubs)]
    
    spiked_pos_ubs = dict(zip(spiked_pos, spiked_ubs))
    
    # 2.2) Spike signals chunk
    if not fully_synth:
        spiked_chunk = spike_chunk(chunk, length, target[:length], breakpts[:length], 
            spiked_pos_ubs, kmer_poremodel, rng=rng, **syn_signal_kwargs)
    
    # 2.3) Spike target
    spiked_target = np.array(target)
    if ubs != ['N']: # If N, then do not insert any UB, just spike DNA with synthetic signals
        spiked_target[spiked_pos] = [ base_map_rev[ub] for ub in spiked_ubs ]
    
    if fully_synth:
        spiked_chunk = sim_target(spiked_target[:length], breakpts[:length], 
            kmer_poremodel, rng=rng, **syn_signal_kwargs)
    
    return spiked_chunk, spiked_target