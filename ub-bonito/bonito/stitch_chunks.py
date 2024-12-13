#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_MAP = ['N','A','C','G','T','X','Y']

# from numpy.lib.stride_tricks import sliding_window_view # New in version 1.20.0 
def sliding_window_view(array, window_len):
    return np.stack([array[st:st+window_len] for st in range(len(array)-window_len+1)])

def read_ctc_data(ctc_dir, bkps=False):
    chunks = np.load(os.path.join(ctc_dir, 'chunks.npy'), mmap_mode='r')
    targets = np.load(os.path.join(ctc_dir, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(ctc_dir, "reference_lengths.npy"), mmap_mode='r')
    
    if bkps:
        bkps = np.load(os.path.join(ctc_dir, "breakpoints.npy"), mmap_mode='r')
        return chunks, targets, lengths, bkps
    return chunks, targets, lengths

def convert_target_to_string(target, base_map=BASE_MAP):
    return ''.join([ base_map[i] for i in target ])

def load_kmers_weight(ctc_dir, kmer_len=6):
    kmer_cnt_filepath = os.path.join(ctc_dir, f'kmer_count-len_{kmer_len}.csv')
    
    #### a) Balanced weights single 6-mer
    # kmers_cnt_df = pd.read_csv(kmer_cnt_filepath, index_col='kmer')
    # kmers_weight_df = kmers_cnt_df.sum() / (len(kmers_cnt_df)*kmers_cnt_df)
    
    #### b) Balanced weights considering 11-mer (6*6-mer)
    kmers_cnt_df = pd.read_csv(kmer_cnt_filepath)
    for i in range(kmer_len):
        kmers_cnt_df[f'kmer_N_{i}'] = kmers_cnt_df.kmer.apply(lambda x: x[:i]+'N'+x[i+1:])
    melt_kmers_cnt_df = kmers_cnt_df.melt(id_vars=['cnt'], value_name='N_kmer', 
        value_vars=[f'kmer_N_{i}' for i in range(kmer_len)]).drop(columns=['variable'])
    n_kmers_cnt_df = melt_kmers_cnt_df.groupby('N_kmer').cnt.sum()
    kmers_weight_df = n_kmers_cnt_df.sum() / (len(n_kmers_cnt_df)*n_kmers_cnt_df)
    
    kmers_weight_df **= 2 # Making weights even more accentuated
    
    return kmers_weight_df

def choose_positions_weighted(target, n_pos, kmers_weight_df, pad=5, ubs_pos=None,
                              rng=np.random, kmer_len=6):
    #### a) Single 6-mer consideration
    # tar_kmers = sliding_window_view(target, kmer_len)
    # tar_kmers = [ convert_target_to_string(k) for k in tar_kmers ]
    # tar_weights = kmers_weight_df.loc[tar_kmers].squeeze().array
    
    #### b) All 11-mer (6*6-mer) consideration
    target_str = convert_target_to_string(target)
    win_size = 2*kmer_len - 1
    kmers_per_pos = []
    for win_pos in range(len(target)-win_size+1):
        win_kmer = target_str[win_pos:win_pos+win_size]
        win_kmer = win_kmer[:kmer_len-1] + 'N' + win_kmer[kmer_len:]
        kmers = [win_kmer[p:p+kmer_len] for p in range(kmer_len)]
        # kmers = [win_kmer[p:p+kmer_len] for p in range(2, kmer_len-2)] # Speed-up 10s/20s
        kmers_per_pos.append(kmers)
    #### Slow .loc operation
    ### imprv1: .loc separatedly (5x speed-up)
    ### imprv2: .get (20s speed-up) and allows default 0 if kmer has UB (not in weights)
    # tar_weights = [ np.mean([kmers_weight_df.get(k,0) for k in ks]) for ks in kmers_per_pos ]
    # tar_weights = [ np.min([kmers_weight_df.get(k,0) for k in ks]) for ks in kmers_per_pos ]
    ### Geometric mean
    tar_weights = [ np.prod([kmers_weight_df.get(k,0) for k in ks])**(1.0/len(ks)) for ks in kmers_per_pos ]
    ### Harmonic mean
    # tar_weights = [ 1.0 / np.average(1.0 / np.array([kmers_weight_df.get(k,0) for k in ks])) for ks in kmers_per_pos ]
    
    tar_weights = np.pad(tar_weights, kmer_len-1) # Edges should be 0
    # return np.linspace(10,len(target)-10, n_pos).astype(int)
    
    valid_pos_mask = np.full(len(target), True)
    ### Avoiding first+last 10 bases, as DTW segmentation at edges is messier
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
        
        ### Probs. need to sum to 1, so retrieve weights then norm
        valid_weights = tar_weights[valid_pos] # b)
        valid_weights = valid_weights/valid_weights.sum()
        pos = rng.choice(valid_pos, 1, p=valid_weights)[0]
        #### TODO if kmer is repeated, resample?
        
        st, en = max(0, pos-pad), pos+pad+1
        valid_pos_mask[st:en] = False
        spiked_pos.append(pos)
    spiked_pos.sort()
    return spiked_pos

def choose_positions(length, n_pos, pad=5, rng=np.random, ubs_pos=None):
    valid_pos_mask = np.full(length, True)
    ### Avoiding first+last 10 bases, as DTW segmentation at edges is messier
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

def slice_xna(ctc_dir, stitch_mode, edge_len=5, kmer_len=6, verbose=False, 
              include_chunks=False, max_kmer_cnt=100, base_map=BASE_MAP):
    if stitch_mode == 'mixed':
        xna_slices_dict = {}
        for mode in ['per_slice','per_kmer']:
            xna_slices_dict[mode] = slice_xna(ctc_dir, mode, 
                edge_len=edge_len, kmer_len=kmer_len, verbose=verbose, 
                include_chunks=include_chunks)
        return xna_slices_dict
    
    chunks, targets, lengths, bkps = read_ctc_data(ctc_dir, bkps=True)
    bkps = bkps.astype(int) # Change from uint16 to int32 (to allow negative computations)
    
    if verbose: print(f"> Slicing {stitch_mode}...")
    cnt_close_edge = 0
    xna_slices_info = []
    for read_idx, (chunk, target, length, bkp) in enumerate(
            tqdm(zip(chunks, targets, lengths, bkps), total=len(lengths), 
                 leave=False, desc=f"Slicing {stitch_mode}")):
                # disable=(not verbose))):
        target, bkp = target[:length], bkp[:length]
        ub_pos = np.argwhere(target > 4).squeeze(-1)[0]
        
        if not edge_len < ub_pos < length - edge_len:
            cnt_close_edge += 1
            continue

        slice_target = target[ub_pos-kmer_len+1:ub_pos+kmer_len]
        slice_bkp = bkp[ub_pos-kmer_len:ub_pos+1] # only need edges?
        
        ### Discard read if one of the kmers has too many signals
        kmer_cnts = np.diff(slice_bkp)
        if max_kmer_cnt and kmer_cnts.max() > max_kmer_cnt:
            continue
        
        if stitch_mode == 'per_kmer':
            #### per_kmer
            for kmer_idx, kmer in enumerate(sliding_window_view(slice_target, kmer_len)):
                kmer_str = ''.join([ base_map[i] for i in kmer])
                kmer_bkp = slice_bkp[kmer_idx:kmer_idx+2]
                
                info = dict(
                    read_idx = read_idx,
                    kmer = kmer_str,
                    ub = base_map[slice_target[5]],
                    # chunk_len = len(slice_chunk),
                    chunk_len = kmer_bkp[-1]-kmer_bkp[0],
                    
                    # bkps = slice_bkp[1:] - slice_bkp[0],
                    slice_st = kmer_bkp[0], 
                    slice_en = kmer_bkp[-1],
                    # kmer_cnts = np.diff(kmer_bkp),
                    template = ''.join([ base_map[i] for i in slice_target[:5]]),
                    kmer_ub_pos = kmer_len - kmer_idx - 1,
                )
                if include_chunks:
                    slice_chunk = np.array(chunk[kmer_bkp[0]:kmer_bkp[-1]])
                    # slice_chunk = slice_chunk.tolist(), # Convert from memmap
                    # slice_chunk = np.array(slice_chunk), # less memory bcs keeps float32
                    info['chunk'] = slice_chunk
                
                xna_slices_info.append(info)
        elif stitch_mode == 'per_slice':
            #### per_slice
            info = dict(
                read_idx = read_idx,
                kmer = ''.join([ base_map[i] for i in slice_target[:5]]),
                ub = base_map[slice_target[5]], # String 'X' or 'Y'
                # chunk_len = len(slice_chunk),
                chunk_len = slice_bkp[-1]-slice_bkp[0],
                
                # chunk = slice_chunk,
                # chunk = np.array(slice_chunk),
                # chunk = slice_chunk.tolist(), 
                
                # bkps = slice_bkp[1:] - slice_bkp[0],
                kmer_cnts = np.diff(slice_bkp),
                slice_st = slice_bkp[0], 
                slice_en = slice_bkp[-1],
            )
            if include_chunks:
                slice_chunk = np.array(chunk[slice_bkp[0]:slice_bkp[-1]])
                info['chunk'] = slice_chunk
                
            xna_slices_info.append(info)

    xna_slices_df = pd.DataFrame(xna_slices_info)
    
    if verbose:
        print(f"UBs too close to edge: {cnt_close_edge:,d}")
        print("Kmer counting:")
        print(xna_slices_df.groupby(['kmer','ub']).size().groupby(level='ub').describe().round(1))
        print("Chunk len description:")
        print(xna_slices_df.groupby(['ub']).chunk_len.describe(percentiles=[.25,.75,.90,.95,.99]).round(1))
        
    # xna_slices_df.chunk_len = xna_slices_df.chunk_len.astype(np.uint16)
    # xna_slices_df.info(memory_usage='deep')
    # print(xna_slices_df.memory_usage(deep=True))
    
    if stitch_mode == 'per_kmer':
        # xna_slices_df = xna_slices_df.set_index(['ub','kmer','read_idx']).sort_index()
        ### Grouping by UB and XNA1024 template (2x1024) to speed-up candidates' search
        # xna_slices_df = xna_slices_df.set_index(['ub','template','kmer','read_idx']).sort_index()
        xna_slices_df = xna_slices_df.set_index(['ub','template','kmer_ub_pos','kmer','read_idx']).sort_index()
        # xna_slices_df.info(memory_usage='deep') ### Print df mem size
        # xna_slices_df = xna_slices_df.groupby(level=['ub','template'])
        xna_slices_df = xna_slices_df.groupby(level=['ub','template','kmer_ub_pos'])
        xna_slices_df.indices #### forcing groupby resolution
    elif stitch_mode == 'per_slice':
        xna_slices_df = xna_slices_df.set_index(['ub','kmer','read_idx']).sort_index()
        # xna_slices_df.info(memory_usage='deep') ### Print df mem size
    
    return xna_slices_df

def prepare_slice_chunk(slice_chunk, ins_len, kmer_cnts, verbose=False):
    slice_len = len(slice_chunk)
    if verbose: print(f"Preparing sliced chunk: {slice_len=}, {ins_len=}")
    if slice_len < ins_len: # Slice chunk too short, need to add signals
        if verbose: print("Adding signals...", ins_len-slice_len)
        ### a) interpolate values in between (might create unrealistic vals)
        xp = np.linspace(0, ins_len-1, num=slice_len, dtype=int)
        ### b) Avoid interpolation between different kmers
        left_xp, offset = 0, 0
        new_xp = []
        for cnt in kmer_cnts[:-1]:
            right_xp = int(np.floor(xp[offset+cnt-1:offset+cnt+1].mean()))
            # kmer_xp = np.linspace(left_xp, right_xp, num=cnt, dtype=int)
            kmer_xp = np.linspace(left_xp, right_xp, num=cnt).round().astype(int) # more central
            new_xp += kmer_xp.tolist()
            left_xp = right_xp+1
            offset += cnt
        # kmer_xp = np.linspace(left_xp, ins_len-1, num=kmer_cnts[-1], dtype=int)
        kmer_xp = np.linspace(left_xp, ins_len-1, num=kmer_cnts[-1]).round().astype(int)
        new_xp += kmer_xp.tolist()
        xp = np.asarray(new_xp)
        slice_chunk = np.interp(x=np.arange(ins_len), xp=xp, fp=slice_chunk)
    elif slice_len > ins_len: # Slice chunk too long, need to rmv signals
        if verbose: print("Removing signals...", slice_len - ins_len)
        n_rmv = slice_len - ins_len
        rmv_idxs = np.linspace(0, slice_len-1, num=n_rmv, dtype=int)
        keep_mask = np.ones(slice_len, dtype=bool)
        keep_mask[rmv_idxs] = False
        slice_chunk = np.asarray(slice_chunk)[keep_mask]
    
    return slice_chunk

def read_and_slice_chunk(ctc_dir, chunk_id, slice_st, slice_en):
    chunks = np.load(os.path.join(ctc_dir, 'chunks.npy'), mmap_mode='r')
    chunk_slice = chunks[chunk_id, slice_st:slice_en]
    return chunk_slice

def read_and_slice_chunks(ctc_dir, slices_idxs):
    chunks = np.load(os.path.join(ctc_dir, 'chunks.npy'), mmap_mode='r')
    # for chunk_id, slice_st, slice_en in slices_idxs:
    #     chunk_slices = chunks[chunk_id, slice_st:slice_en]
    chunk_slices = [chunks[chunk_id, slice_st:slice_en] 
                    for chunk_id, slice_st, slice_en in slices_idxs ]
    return chunk_slices

def transform_chunk(chunk, permute_win_size=0, noise_std=0, noise_mode='single', 
                    rng=np.random):
    ### Data Augmentation method
    ## 1) permute
    ## 2) Add noise
    # transf_chunk = chunk.copy() # transf_chunk[:] =
    transf_chunk = chunk # transf_chunk = 
    
    if permute_win_size and permute_win_size > 0:
        num_wins = max( round(len(chunk)/permute_win_size) , 1)
        transf_chunk = np.hstack([rng.permutation(spl) for spl in 
                                     np.array_split(transf_chunk, num_wins)])
    
    if noise_std and noise_std > 0:
        from scipy.stats import truncnorm
        std_trunc = 3
        
        if noise_mode == 'single':
            noise = truncnorm.rvs(-std_trunc, std_trunc, scale=noise_std, 
                                  size=len(chunk), random_state=rng)
        elif noise_mode == 'single_variable':
            noise = truncnorm.rvs(-std_trunc, std_trunc, 
                                  scale=rng.uniform(0, noise_std), 
                                  size=len(chunk), random_state=rng)
        elif noise_mode == 'block_add':
            noise_addend = rng.uniform(-noise_std, noise_std)
            noise = np.repeat(noise_addend, len(chunk))
        elif noise_mode == 'block_mult':
            noise_mult = rng.uniform(-noise_std, noise_std)
            noise = transf_chunk * noise_mult
        else:
            raise ValueError(f"Invalid noise mode = {noise_mode}")
            
        transf_chunk = transf_chunk + noise
    
    return transf_chunk

def stitch_read_per_kmer(single_ctc_data, xna_slices_df, ubs, prop_ubs, var_prop_ubs=None,
                         cand_sample_size=10, verbose=False, rng=np.random, kmer_len=6, 
                         xna_ctc_dir=None, kmers_weight_df=None, base_map=BASE_MAP,
                         pad=5, **transf_kwargs):
    chunk, target, length, bkp = single_ctc_data # Use tuple to simplify parallelization
    
    stitch_target = np.array(target)
    stitch_chunk = np.array(chunk)
    target = target[:length]
    bkp = np.array(bkp[:length])
    #### TODO stitch_bkp is needed, because I am changing kmer reps inside sliced region
    base_map_rev = dict(zip(base_map, range(len(base_map))))
    
    success = False
    if var_prop_ubs is not None and var_prop_ubs > 0:
        prop_ubs = rng.uniform(prop_ubs-var_prop_ubs, prop_ubs+var_prop_ubs)
    ubs_pos = np.argwhere(target[:length]>4)[:,0]
    n_pos = round(length*prop_ubs) - len(ubs_pos) # Take into account quantity of UBs already present
    n_pos = max(n_pos, 1) # Force to have at least one UB added
    
    if kmers_weight_df is None:
        insert_positions = choose_positions(length, n_pos, rng=rng, ubs_pos=ubs_pos, pad=pad)
    else:
        insert_positions = choose_positions_weighted(
            target, n_pos, kmers_weight_df, ubs_pos=ubs_pos, rng=rng, pad=pad)
    
    for insert_pos in insert_positions:
        ins_st, ins_en = bkp[[insert_pos-kmer_len,insert_pos]]
        ins_len = ins_en-ins_st
        
        slice_target = np.array(target[insert_pos-kmer_len+1:insert_pos+kmer_len])
        ub = rng.choice(ubs)
        slice_target[kmer_len-1] = base_map_rev[ub]
        
        # slice_tar_str = ''.join([ base_map[i] for i in slice_target])
        slice_chunks, kmers_cnts = [], []
        ins_kmers = sliding_window_view(slice_target, kmer_len)
        ins_kmer_reps = np.diff(bkp[insert_pos-kmer_len:insert_pos+1])
        # for kmer in ins_kmers:
        # for kmer, kmer_rep in zip(ins_kmers, ins_kmer_reps):
        for kmer_idx, (kmer, kmer_rep) in enumerate(zip(ins_kmers, ins_kmer_reps)):
            ins_target = ''.join([ base_map[i] for i in kmer])
            
            kmer_ub_pos = kmer_len - kmer_idx - 1
            tpl = ins_target[kmer_ub_pos+1:] + ins_target[:kmer_ub_pos]
            
            try: # Catch KeyError when looking for perfect match
                # candidates = xna_slices_df.xs((ub,ins_target), level=('ub','kmer'), drop_level=False)
                ### Speed-up attempt: higher hierarchy
                # candidates = xna_slices_df.xs(
                #     (ub,tpl,ins_target), level=('ub','template','kmer'), drop_level=False)
                # candidates = xna_slices_df.get_group((ub,tpl)).xs(
                    # (ins_target,), level=('kmer',), drop_level=False)
                ### speed-up attempt v2: even higher hierarchy + wo/ .xs
                candidates = xna_slices_df.get_group((ub,tpl,kmer_ub_pos))
                
                ### Step-wise: Much slower!
                # candidates = (xna_slices_df.xs((ub,), level=('ub',), drop_level=False)
                #                            .xs((tpl,), level=('template',), drop_level=False)
                #                            .xs((ins_target,), level=('kmer',), drop_level=False))
            except KeyError: # No valid candidate for this kmer
                slice_chunks = kmers_cnts = []
                break # Stop, failed to find one of the ub kmers needed
            #     candidates = xna_slices_df.loc[[]]
            # if candidates.empty:
            #     slice_chunks = kmers_cnts = []
            #     break # Stop, failed to find one of the ub kmers needed
            
            #### TODO try sampling less candidates to speed-up (less accurate?)
            if cand_sample_size > 1:
                # candidates = candidates.sample(min(len(candidates), cand_sample_size), random_state=rng)
                # # chosen_xna_slice = candidates.sort_values('chunk_len', key=lambda x: abs(x-kmer_rep)).iloc[0]
                # idx_min_len_diff = candidates.chunk_len.apply(lambda x: abs(x-kmer_rep)).argmin()
                
                ### Numpy sampling is slightly faster (this also shuffles?)
                cand_nums = rng.choice(len(candidates), 
                    size=min(len(candidates), cand_sample_size), replace=False)
                cand_lens = candidates.chunk_len.to_numpy()[cand_nums]
                idx_min_len_diff = cand_nums[np.absolute(cand_lens-kmer_rep).argmin()]
                
                chosen_xna_slice = candidates.iloc[idx_min_len_diff]
            else:
                # chosen_xna_slice = candidates.sample(1, random_state=rng).squeeze()
                cand_num = rng.choice(len(candidates), size=1)[0]
                chosen_xna_slice = candidates.iloc[cand_num]
            
            #### Read chunk from xna_ctc_dir
            if xna_ctc_dir is None:
                chunk_slice = chosen_xna_slice.chunk
                if transf_kwargs:
                    chunk_slice = transform_chunk(chunk_slice, rng=rng, **transf_kwargs)
            else:
                chunk_id = chosen_xna_slice.name[-1]
                # slice_st, slice_en = chosen_xna_slice[['slice_st','slice_en']] # Too slow
                # slice_st, slice_en = chosen_xna_slice.to_numpy()[[1,2]]
                slice_st = chosen_xna_slice.slice_st
                slice_en = chosen_xna_slice.slice_en
                # chunk_slice = read_and_slice_chunk(xna_ctc_dir, chunk_id, slice_st, slice_en)
                chunk_slice = (chunk_id, slice_st, slice_en)
            # continue
            
            slice_chunks.append(chunk_slice)
            # kmers_cnts.append(chosen_xna_slice.kmer_cnts)
            kmers_cnts.append(chosen_xna_slice.chunk_len)
        
        if slice_chunks == []:
            if verbose: print("  > Skipping insert_pos: no valid kmer candidate...")
            continue # Skip because it is too close to previous insertion
        
        if xna_ctc_dir is not None:
            slice_chunks = read_and_slice_chunks(xna_ctc_dir, slice_chunks)
            if transf_kwargs:
                slice_chunks = [transform_chunk(chunk_slice, rng=rng, **transf_kwargs)
                                for chunk_slice in slice_chunks]
        
        slice_chunk = np.concatenate(slice_chunks)
        # kmer_cnts = np.concatenate(kmers_cnts)
        kmer_cnts = kmers_cnts
        # print(f"{ins_kmer_reps=}"); print(f"{kmer_cnts=}")
        slice_chunk = prepare_slice_chunk(slice_chunk, ins_len, kmer_cnts, verbose=verbose)
        
        stitch_target[insert_pos] = base_map_rev[ub.upper()]
        stitch_chunk[ins_st:ins_en] = slice_chunk
        success = True
    
    ### Skip-TODO move read_and_slice_chunks here, to perform a single call
    # Similar to what I did at stitch v1?!?!
    
    if not success:
        if verbose: print("[WARNING] No stitch executed! No valid kmer or no XNA candidate.")
    
    return stitch_chunk, stitch_target, success
    
    
def stitch_read_per_slice(single_ctc_data, xna_slices_df, ubs, cand_sample_size=10,
                          verbose=False, rng=np.random, kmer_len=6, xna_ctc_dir=None,
                          base_map=BASE_MAP, **transf_kwargs):
    chunk, target, length, bkp = single_ctc_data # Use tuple to simplify parallelization
    stitch_target = np.array(target)
    stitch_chunk = np.array(chunk)
    target = target[:length]
    bkp = bkp[:length]
    #### TODO stitch_bkp is needed, because I am changing kmer reps inside sliced region
    
    tar_kmers = sliding_window_view(target, 11)
    valid_kmers_mask = [ np.array_equal(x[:5], x[6:]) for x in tar_kmers ]
    valid_positions = np.argwhere(valid_kmers_mask).squeeze(-1) + kmer_len - 1
    #### TODO Avoid edge pos?
    valid_positions = valid_positions[valid_positions > kmer_len]
    if verbose: print(f"Valid 1024-like regions: {len(valid_positions)}")
    
    base_map_rev = dict(zip(base_map,range(len(base_map))))
    prvs_insert_pos = -np.inf
    for insert_pos in valid_positions:
        if verbose: print(f"+++++ {insert_pos=} +++++++++++++++++++++++++++++")
        if insert_pos - kmer_len < prvs_insert_pos:
            if verbose: print("  > Skipping insert_pos: too close to prvs insert.")
            continue # Skip because it is too close to previous insertion
        # target[insert_pos-kmer_len+1:insert_pos+kmer_len]
        slice_target = target[insert_pos-kmer_len+1:insert_pos+kmer_len]
        ins_target = ''.join([ base_map[i] for i in slice_target[:5]])
        # ins_target = convert_string_to_target(slice_target[:5])
        if verbose: print(f"  {ins_target=}")
        
        #### flip-coin and choose UB to further filter candidates
        # ub = rng.choice(ubs)
        ### Adding chance to skip valid position
        ### For even base distribution, should consider all four nat bases
        # ub = rng.choice(ubs+['N']) 
        ub = rng.choice(ubs+list('ACGT')) 
        if ub not in ubs:
            ### set prvs_insert_pos, so that read counts as success
            ### Might cause close insert pos to be skipped
            if verbose: print("  > Skipping insert_pos: random pick to preserve sequence.")
            prvs_insert_pos = insert_pos
            continue
        
        try: # Catch KeyError when looking for perfect match
            # candidates = xna_slices_df.xs((ins_target,), level=('kmer',), drop_level=False)
            # candidates = xna_slices_df.xs((ins_target,ub), level=('kmer','ub'), drop_level=False)
            candidates = xna_slices_df.xs((ub,ins_target), level=('ub','kmer'), drop_level=False)
        except KeyError: # No valid candidate for this kmer
            candidates = xna_slices_df.loc[[]]
        
        if candidates.empty:
            if verbose: print("  > Skipping insert_pos: no valid kmer candidate.")
            continue # Skip because it is too close to previous insertion
        
        ins_st, ins_en = bkp[[insert_pos-kmer_len,insert_pos]]
        ins_len = ins_en-ins_st
        
        #### Choosing XNA slice to insert 
        #### TODO try sampling less candidates to speed-up (less accurate?)
        if cand_sample_size > 1:
            # candidates = candidates.sample(min(len(candidates), cand_sample_size), random_state=rng)
            # # chosen_xna_slice = candidates.sort_values('chunk_len', key=lambda x: abs(x-ins_len)).iloc[0]
            # idx_min_len_diff = candidates.chunk_len.apply(lambda x: abs(x-ins_len)).argmin()
            
            ### Numpy sampling is slightly faster
            cand_nums = rng.choice(len(candidates), 
                size=min(len(candidates), cand_sample_size), replace=False)
            cand_lens = candidates.chunk_len.to_numpy()[cand_nums]
            idx_min_len_diff = cand_nums[np.absolute(cand_lens-ins_len).argmin()]
            
            chosen_xna_slice = candidates.iloc[idx_min_len_diff]
        else:
            # chosen_xna_slice = candidates.sample(1, random_state=rng).squeeze()
            cand_num = rng.choice(len(candidates), size=1)[0]
            chosen_xna_slice = candidates.iloc[cand_num]
        
        # read_idx, kmer, ub = chosen_xna_slice.name
        ub, kmer, read_idx = chosen_xna_slice.name
        if verbose: print(f"  {ub=} {read_idx=}")
        
        #### Read from xna_ctc_data
        if xna_ctc_dir is None:
            slice_chunk = chosen_xna_slice.chunk
        else:
            chunk_id = chosen_xna_slice.name[-1]
            slice_st, slice_en = chosen_xna_slice[['slice_st','slice_en']]
            slice_chunk = read_and_slice_chunk(xna_ctc_dir, chunk_id, slice_st, slice_en)
        
        kmer_cnts = chosen_xna_slice.kmer_cnts
        slice_chunk = prepare_slice_chunk(slice_chunk, ins_len, kmer_cnts, verbose=verbose)
        
        stitch_target[insert_pos] = base_map_rev[ub.upper()]
        stitch_chunk[ins_st: ins_en] = slice_chunk
        #### TODO keep track of XNA read_ids used?
        
        prvs_insert_pos = insert_pos
    
    success = prvs_insert_pos != -np.inf
    if prvs_insert_pos == -np.inf:
        if verbose: print("[WARNING] No stitch executed! No valid kmer or no XNA candidate.")
    
    return stitch_chunk, stitch_target, success

def stitch_read(single_ctc_data, xna_slices_df, ubs, stitch_mode='per_slice', 
                prop_ubs=None, var_prop_ubs=None, kmers_weight_df=None, **stitch_kwargs):
    if stitch_mode == 'per_slice':
        stitched_read = stitch_read_per_slice(single_ctc_data, 
            xna_slices_df, ubs, **stitch_kwargs)
    elif stitch_mode == 'per_kmer':
        stitched_read = stitch_read_per_kmer(single_ctc_data, 
            xna_slices_df, ubs, prop_ubs, var_prop_ubs=var_prop_ubs, 
            kmers_weight_df=kmers_weight_df, **stitch_kwargs)
    elif stitch_mode == 'mixed':
        stitched_read = stitch_read_per_slice(single_ctc_data, 
            xna_slices_df['per_slice'], ubs, **stitch_kwargs)
        stitch_chunk, stitch_target, success = stitched_read
        chunk, target, length, bkp = single_ctc_data
        stitch_single_ctc_data = (stitch_chunk, stitch_target, length, bkp)
        stitched_read = stitch_read_per_kmer(stitch_single_ctc_data, 
            xna_slices_df['per_kmer'], ubs, prop_ubs, var_prop_ubs=var_prop_ubs,
            kmers_weight_df=kmers_weight_df,
            **stitch_kwargs)
    else:
        raise ValueError("Invalid choice for stitch_mode:", stitch_mode)
    
    return stitched_read
