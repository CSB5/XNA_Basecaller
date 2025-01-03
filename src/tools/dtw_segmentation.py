#!/usr/bin/env python
import argparse, os, time, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from dtw import dtw, StepPattern
from scipy.signal import medfilt
from itertools import starmap
from functools import partial
import multiprocessing as mp

sys.path.append(os.path.dirname(sys.path[0]))
from misc.data_io import load_kmer_poremodel
from misc.utils import normalize_med_mad_squiggly

DEF_REF_FILEPATH = './ub-bonito/bonito/data/r9.4_450bps.nucleotide.6mer.XNA-Px_Ds.template.model'

def load_args():
    ap = argparse.ArgumentParser(
        description='Generate DTW signal segmentation of input ctc-data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('ctc_dir',
        help='path to ctc-data directory which breakpoints.npy will be generated.',
        type=str)
    
    # Optional arguments
    ap.add_argument('-r','--ref_filepath',
        help="Filepath to reference kmer model. Loads default natural reference if None.",
        default=DEF_REF_FILEPATH, type=str)
    ap.add_argument('-R','--ref_rep',
        help='Repeat value for template during DTW alignment (sets min number tpl matches).',
        default=3, type=int)
    ap.add_argument('-u','--ubs_map', 
        type=list,
        help='Use mapping of Nat. bases to replace the UBs when generating the reference signal.\
              First letter is for X and secod for Y. Ex: --ubs_map AT -> X=A Y=T')
    ap.add_argument('-S','--suffix',
        help="Suffix string to add to breakpoints file",
        # default='',
        type=str)
    ap.add_argument('-n','--naive',
        help="Output naive segmentation, which assumes same repetition for all kmers: chunksize/length",
        action='store_true')
    ap.add_argument('-w','--window_size',
        help='window size to be used with slanted band global constraint. Limit kmer rep.',
        type=int)
    
    ap.add_argument('-p', '--parallel',
        # help="(not) run in parallel.", action='store_false')
        help="run in parallel.", action='store_true')
    ap.add_argument('--n_proc',
        help='Number of processes to run in parallel. (None for No. of CPUs)',
        type=int)
    ap.add_argument('--pool_chunksize',
        help='Chunksize for pool.imap. If None it will use N_chunks/n_proc (high memory usage!).',
        # default=500, 
        type=int)
    
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def print_bkps_stats(lengths, bkps, leave=False, verbose=True):
    bkps_stats = []
    for length, breakpts in tqdm(zip(lengths, bkps), total=len(bkps), leave=leave,
                                 desc='Counting kmer reps'):
        kmer_reps = np.diff(breakpts[:length])
        kmer_reps_stats = pd.Series(kmer_reps).describe()
        mad = np.median(np.absolute(kmer_reps - np.median(kmer_reps)))
        kmer_reps_stats['mad'] = mad
        bkps_stats.append(kmer_reps_stats)
    print()
    
    bkps_stats = pd.concat(bkps_stats, axis=1).T
    if verbose:
        percs = [.25,.75,.9,.95,.99]
        print("Kmer repetitions stats per chunk:")
        # print(bkps_stats[['mean','25%','50%','75%','max']].describe(percs).T.round(1))
        print(bkps_stats[['mad','mean','50%','max']].describe(percs).T.round(1))
        print(bkps_stats[['min']].describe(percs).T.round(1))
    
    return bkps_stats

def get_kmers_model(sequence, kmer_poremodel, k, append=True):
    ### Append tail with ATs (most common for Bonito DNA and not homopolymer)
    if append:
        if sequence[-1] != 'A':
            sequence += 'ATATA'
        else:
            sequence += 'TATAT'
        if k < 6:
            sequence = sequence[:k-6]
    # ### Append tail with CGs (most common and not homopolymer)
    # if append:
    #     if sequence[-1] != 'C':
    #         sequence += 'CGCGC'
    #     else:
    #         sequence += 'GCGCG'
        
    length = len(sequence)
    if length < k:
        kmer_means=list()
        kmer_stdvs=list()
        [kmer_means.extend((float(90.2083),)*1) for i in range(length)]
        [kmer_stdvs.extend((float(2.0),)*1) for i in range(length)]
    else:
        kmers = [sequence[i:i + k] for i in range(0, length - k + 1)]
        ### [off] Append tail by repeating last kmer
        # kmers += kmers[-1:]*(k-1)
        kmer_means, kmer_stdvs = zip(*[kmer_poremodel[kmer] for kmer in kmers])
        kmer_means = list(kmer_means)
        kmer_stdvs = list(kmer_stdvs)
        ### [off] Append tail with arbitrary fixed values
        # [kmer_means.extend((float(90.2083),)*1) for i in range(k-1)]
        # [kmer_stdvs.extend((float(2.0),)*1) for i in range(k-1)]
    
    # kmer_means = np.nan_to_num(np.array(kmer_means)) #remove to_num later, temp
    # kmer_stdvs = np.nan_to_num(np.array(kmer_stdvs))

    return kmer_means,kmer_stdvs

def segment_read(chunk, length, target, kmer_poremodel, ref_rep=2, smooth_val=1, 
                 ubs_map=None, window_size=None, ref_filepath=None, kmer_len=6):
    
    if kmer_poremodel is None:
        kmer_poremodel = load_kmer_poremodel(ref_filepath)
    
    base_map = ['N','A','C','G','T','X','Y']
    target = target[:length]
    
    if ubs_map is not None:
        target = target.copy()
        base_map_rev = dict(zip(base_map,range(len(base_map))))
        target[target==5] = base_map_rev[ubs_map[0]] # G
        target[target==6] = base_map_rev[ubs_map[1]] # T
        
    target_str = ''.join([ base_map[i] for i in target])
    
    #### Generate ref signal
    mean_result, std_result = get_kmers_model(target_str, kmer_poremodel, k=kmer_len)
    ref_signal = normalize_med_mad_squiggly(mean_result, std_result)
    
    if smooth_val > 1:
        if chunk.dtype == np.float16:
            chunk = np.asarray(chunk, dtype=float)
        chunk = medfilt(chunk, smooth_val)

    #### Run DTW Align
    if window_size is None:
        window_type=None; window_args={}
    else:
        window_type = 'slantedband'
        ### v2 - 
        balanced_mean = len(chunk)/length
        window_args = {'window_size': balanced_mean*window_size}
        ### v1 - for asymmetric step w/ ref_rep=6
        # window_args = {'window_size': window_size//ref_rep}
        # print(window_size, ref_rep, window_args)
        # window_type = 'itakura'; window_args={} # Fail
    
    ## My step_pattern (disallow skipping refs)
    mx_asymmetric = np.array([[ 1.,  1.,  0., -1.],[ 1.,  0.,  0.,  1.],
                              [ 2.,  1.,  1., -1.],[ 2.,  0.,  0.,  1.],])
    my_asymmetric = StepPattern(mx_asymmetric, "N")
    
    reference = ref_signal
    if ref_rep > 1:
        reference = np.repeat(reference, ref_rep)
    
    try:
        alignment = dtw(chunk, reference, keep_internals=True, 
                        # step_pattern='asymmetric', # Single qry use, might skip refs
                        step_pattern=my_asymmetric, # Single qry use, no skip refs
                        window_type=window_type, window_args=window_args)
        # qry_indices = alignment.index1 # matched elements: indices in query
        ref_indices = alignment.index2 # corresponding mapped indices in reference
    except ValueError as inst:
        if inst.args[0] == 'No warping path found compatible with the local constraints':
            #### Baseline (naive stitch strategy)
            chunksize = chunk.shape[-1]
            reps_per_ref = np.full(length, chunksize//length)
            reps_per_ref[:chunksize%length] += 1
            assert reps_per_ref.sum() == chunksize
            align_bkps = np.cumsum(reps_per_ref)
            return align_bkps, False
        else:
            raise inst
        
    if ref_rep > 1:
        ref_indices = ref_indices // ref_rep
    
    reps_per_ref = np.zeros(length, int)
    for evt_idx in ref_indices: reps_per_ref[evt_idx] += 1
    align_bkps = np.cumsum(reps_per_ref)
    
    return align_bkps, True

def star_segment_read(input_tuple, **kwargs):
    return segment_read(*input_tuple, **kwargs)

def dtw_segmentation(ctc_dir, ref_rep=3, smooth_val=1, window_size=None, 
                     ref_filepath=None, ubs_map=None, naive=False, suffix=None,
                     parallel=True, n_proc=0, pool_chunksize=None, 
                     overwrite=False, debug=False, ):
    print(f"> {ctc_dir=}")
    print(f"> {ref_filepath=}")
    print(f"> {ref_rep=} | {window_size=}")
    print(f"> {smooth_val=}")
    print(f"> {ubs_map=}")
    print(f"> {naive=}")
    print(f"> {parallel=}")
    if not naive:
        output_file = 'breakpoints'
        # # output_file += '-dtw_align'
        # output_file += f'-ref_rep_{ref_rep}'
        # output_file += '' if window_size is None else f"-win_siz_{window_size:03d}"
        # output_file += '' if smooth_val == 1 else f'-smooth_val_{smooth_val}'
        # output_file += '' if ubs_map is None else f"-ubs_map_{''.join(ubs_map)}"
    else:
        output_file = 'breakpoints-naive'
        
    output_file += '.npy' if suffix is None else f'-{suffix}.npy'
    print(f"> {output_file=}")
    output_filepath = os.path.join(ctc_dir, output_file)
    
    if os.path.exists(output_filepath) and not overwrite:
        print("[WARNING] Skipping because output file already exist:\n"+output_filepath)
        return

    chunks_np = np.load(os.path.join(ctc_dir, 'chunks.npy'), mmap_mode='r')
    targets = np.load(os.path.join(ctc_dir, "references.npy"))
    lengths = np.load(os.path.join(ctc_dir, "reference_lengths.npy"))
    
    ### DTW Align params
    segment_read_kwargs = dict(kmer_poremodel=load_kmer_poremodel(ref_filepath), 
        ref_rep=ref_rep, window_size=window_size, smooth_val=smooth_val, ubs_map=ubs_map)
    
    if debug:
        debug_len = 50
        targets = targets[:debug_len]
        lengths = lengths[:debug_len]
        print(f"[WARNING] Debugging: {len(targets)=}")
    
    if not naive:
        loop_input = zip(chunks_np, lengths, targets)
        if not parallel:
            partial_segment_read = partial(segment_read, **segment_read_kwargs)
            ret_tuple = list(starmap(partial_segment_read, tqdm(loop_input,total=len(targets))))
        else:
            with mp.Pool(n_proc) as pool:
                if pool_chunksize is None or pool._processes*pool_chunksize > len(targets):
                    pool_chunksize = np.ceil(len(targets)/pool._processes).astype(int)
                    print("Re-adjusted chunksize because it was bigger than total iteration")
                    
                print(f"> Parallelization: {pool._processes} procs x {pool_chunksize}",
                      f"chunksize ({pool._processes*pool_chunksize} Sum CS)...")
                partial_segment_read = partial(star_segment_read, **segment_read_kwargs)
                ret_tuple = list(tqdm(pool.imap(partial_segment_read, loop_input, 
                                                chunksize=pool_chunksize),
                                      total=len(targets)))
    else:
        #### Naive segmentation: Assumes uniform signal repetition for all kmers
        chunksize = chunks_np.shape[-1]
        ret_tuple = []
        for length in lengths:
            reps_per_ref = np.full(length, chunksize//length)
            reps_per_ref[:chunksize%length] += 1
            # assert reps_per_ref.sum() == chunksize
            align_bkps = np.cumsum(reps_per_ref)
            assert align_bkps[-1] == chunksize # Sanity-check
            ret_tuple.append((align_bkps, True))
    
    all_align_bkps, success = zip(*ret_tuple)
    
    # failed_aligns = np.logical_not(success).sum()
    # print(f"{failed_aligns=}")
    
    bkps = np.zeros_like(targets, dtype=np.uint16)
    for idx, bkp in enumerate(all_align_bkps): bkps[idx, :len(bkp)] = bkp
    
    print("Saving file:", output_filepath)
    np.save(output_filepath, bkps)
    
    # print_bkps_stats(lengths, bkps)
    
    return bkps, success
    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())
    print('> Starting dtw_segmentation - ', time.asctime( time.localtime(time.time()) ))
    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    dtw_segmentation(**args)
    print('\n> Finished dtw_segmentation -', time.asctime( time.localtime(time.time()) ))
