#!/usr/bin/env python
import argparse, sys, os, time
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas() # Enables DataFrame.progress_apply()
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(sys.path[0]))
from misc.utils import (print_reads_stats, get_tar_reads_count, compute_all_error_rates_paf, 
    add_barcode_info, compute_read_matches, polish_target_matches, reverse_complement)
from misc.data_io import (read_multiple_pafs, read_sam, read_tsv, get_read_qual, 
    get_read_seq)
from misc.xna_refs import XNA_refs, EXP_REF_MAP, REF_EXP_MAP, VALID_REFS

def load_args():
    ap = argparse.ArgumentParser(
        description='Analyzes alignment/mapping (.paf), counts reads per target.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('exp_name',
        help='Name of experiment or reference name. Ex: POC, CPLX',
        type=str)
    ap.add_argument('paf_filepath',
        help='Filepath to alignment/mapping file (.paf).',
        nargs='+', type=str)
    
    ap.add_argument('-R', '--reads_filepath',
        help='Filepath to reads file (.fasta/.fastq).',
        type=str)
    
    ap.add_argument('-t','--targets_list_filepath',
        help='Filepath to list of targets ids to filter demux.',
        type=str)
    ap.add_argument('-i', '--include_list_filepath', 
        help="path to file with list of read ids to be included (rest will be removed)")

    ap.add_argument('-c','--count_tar_read',
        help='Whether to count number of reads per target.',
        action='store_true')
    
    ap.add_argument('-r','--min_reads_count',
        help='Minimum reads count per template to report as insuficcient.',
        # default=100,
        type=int)
    
    ap.add_argument('-d','--max_bc_dist',
        help='Maximum barcode distance to filter reads.',
        type=int)
        
    ap.add_argument('-p','--analyze_seq_perf',
        help='Whether to analyze sequencing performance (summarized info).',
        action='store_true')
        
    ap.add_argument('-g','--groupby_strand',
        help='Whether to group sequencing performance by strand.',
        action='store_true')
        
    ap.add_argument('-D','--save_detailed_perf',
        help='Whether to save sequencing performance detailed by target and strand.',
        action='store_true')
        
    ap.add_argument('-q','--q_score_print',
        help='Compute read mean quality scores and print.',
        action='store_true')
        
    ap.add_argument('-s','--save_res_summ',
        help='Whether to save sequencing performance summary.',
        action='store_false')
    
    ap.add_argument('--save_perf_per_read',
        help='Whether to save sequencing performance per read.',
        action='store_true')
    
    ap.add_argument('--save_confusion_matrix',
        help='Compute and save bps confusion matrix over full-length aligned reads.',
        action='store_true')
        
    ap.add_argument('-S','--only_strand',
        help='Only use selected strand to compute performance.',
        choices=['F','R'],
        type=str)
    ap.add_argument('-u','--ubs',
        help='Which UBs to be analyzed. Proxy to define strand value.',
        choices=['X','Y','XY'], default='XY',
        type=str)
    
    ap.add_argument('-v','--verbose',
        help='Whether to verbose statistics on target cover, read lens and start.',
        # action='store_true')
        action='count', default=0)
        
    ap.add_argument('-Q','--quiet',
        help='Skip printing some results (per tpl, per pos+tpl, etc).',
        action='store_true')
    
    # Optional arguments
    ap.add_argument('--debug',
        help="debug mode, using much less reads to speed-up",
        action='store_true')
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def compute_stats_error_rate(error_rate, x_positions, kmer_len=6, max_dist=10):
    """
    Slice error rates by grouping bases into different criterias.
    Ex: only_ub, no_ub, outside_ub_area, inside_ub_area, ub_and_ub_area, 
        dist_ub_d-D, dist_ub_d-D+

    Parameters
    ----------
    error_rate : TYPE
        DESCRIPTION.
    x_positions : TYPE
        DESCRIPTION.
    kmer_len : int, optional
        DESCRIPTION. The default is 6.
    max_dist : int, optional
        Maximum distance to create labels dist_ub_d-D. The default is 10.

    Raises
    ------
    ValueError
        x_positions is empty (e.g. []).

    Returns
    -------
    error_rates_cuts : dict of arrays
        Each key represents a possible label for the bases and the value is the
        error rate array with all base positions that fit this label.
        {'only_ub': array([100.]), 'no_ub': array([79.12119064, ...]),
         'inside_ub_area': array([ 2.64351524,  3.79872431,...]),
         'dist_ub_d-1': array([31.48830617, 66.79659816]), 
         'dist_ub_d-2': array([13.3664068 , 24.28773919]),
         ...
         'dist_ub_d-11+': array([14.12912913, 11.17117117, 10.88588589, ...])}
    """
    
    if len(x_positions) == 0:
        raise ValueError("x_positions is empty: "+str(x_positions))
    
    error_rates_cuts = {}
    error_rate = np.asarray(error_rate)
    
    no_ub_mask = np.array([True]*len(error_rate))
    ub_influence_mask = np.array([False]*len(error_rate))
    for x_pos in x_positions:
        x_pos_slice = slice(x_pos + 1 - kmer_len, x_pos + kmer_len)
        ub_influence_mask[x_pos_slice] = True
    for x_pos in x_positions:
        no_ub_mask[x_pos] = False
        ub_influence_mask[x_pos] = True
    
    
    error_rates_cuts['only_ub'] = error_rate[~no_ub_mask]
    
    error_rates_cuts['no_ub'] = error_rate[no_ub_mask]
    
    error_rates_cuts['outside_ub_area'] = error_rate[~ub_influence_mask]

    ### error rates inside UB influence (but ignoring UB error rate)
    error_rates_cuts['inside_ub_area'] = error_rate[ub_influence_mask & no_ub_mask]
    
    error_rates_cuts['ub_and_ub_area'] = error_rate[ub_influence_mask]

    ### error rates by distance to UB
    positions = np.arange(len(error_rate))
    x_pos_distances = np.array([ min([ abs(x_pos - p) for x_pos in x_positions ]) 
                       for p in positions])
    
    # print(np.array([positions, x_pos_distances, no_ub_mask, ub_influence_mask]).T)
    # print(np.array([positions, x_pos_distances, ub_influence_mask & no_ub_mask]).T)

    # max_dist = kmer_len + 1
    for dist in range(1, max_dist+1):
        dist_mask = (x_pos_distances == dist)
        error_rates_cuts['dist_ub_d-'+str(dist)] = error_rate[dist_mask]
        
    ### Adding label aggregating remaining dists
    dist_mask = (x_pos_distances >= max_dist+1)
    error_rates_cuts['dist_ub_d-'+str(max_dist+1)+'+'] = error_rate[dist_mask]
    
    return error_rates_cuts

def aggregate_stats_error_rate(exp, strand, error_rates_dir, selected_targets=None,
                               file_tpl='{}-{}-{}.npy', error_rates_dict=None,
                               max_dist=10):
    """
    Concatenates labeled error rates from all targets into a single DataFrame.
    Base position error rates are labeled according to different criterias:
        only_ub, no_ub, outside_ub_area, inside_ub_area, ub_and_ub_area, dist_ub_d-D
    

    Parameters
    ----------
    exp : str
        DESCRIPTION.
    strand : str
        F/R.
    error_rates_dir : TYPE
        DESCRIPTION.
    selected_targets : TYPE, optional
        DESCRIPTION. The default is None.
    file_tpl : TYPE, optional
        DESCRIPTION. The default is '{}-{}-{}.npy'.
    error_rates_dict : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    error_rates_df : pd.DataFrame
        DataFrame with a row for each base position error rate plus repetition.
        ex:     target_id      label  error_rates  is_pc  num_ubs strand type
        0       XNA03        only_ub   100.000000  False        1      R  XNA
        1       XNA03          no_ub    12.285456  False        1      R  XNA
        2       XNA03          no_ub     9.852454  False        1      R  XNA
        ...
        649     XNA03  dist_ub_d-11+     1.006006   True        0      R   PC

    """
    # Ex file_tpl: XNA01-F-pc.npy, XNA01-F-xna.npy
    ref_name = EXP_REF_MAP[exp]
    refs_info = XNA_refs(ref_name)

    if selected_targets is None:
        # selected_targets = refs_info.xna_targets_id
        selected_targets = refs_info.targets_id
    
    target_lst = []
    label_lst = []
    error_lst = []
    is_pc_lst = []
    num_ubs_lst = []

    for target_id in selected_targets:
        is_pc = target_id.startswith('PC')
            
        xna_target_id = (refs_info.get_complement_target_id(target_id) 
                         if is_pc else target_id)
        
        if error_rates_dict is None:
            error_filename = file_tpl.format(xna_target_id, strand, 
                                             'pc' if is_pc else 'xna')
            error_filepath = os.path.join(error_rates_dir, error_filename)
            error_rate = np.load(error_filepath)
        else:
            error_rate = error_rates_dict[(target_id,strand)]
        
        x_positions = (refs_info.x_pos[xna_target_id] if strand == 'F' else
                       refs_info.x_pos_rev[xna_target_id])
        
        error_rates_cuts = compute_stats_error_rate(error_rate, x_positions, 
                                                    max_dist=max_dist)
        
        
        # print(target_id, is_pc)
        for key, values in error_rates_cuts.items():
            error_lst += list(values)
            label_lst += [key] * len(values)
            target_lst += [xna_target_id] * len(values)
            is_pc_lst += [is_pc] * len(values)
            num_ubs_lst += [0 if is_pc else len(x_positions)] * len(values)
    
    error_rates_df = pd.DataFrame({'target_id': target_lst,
                                   'label': label_lst,
                                   'error_rates': error_lst,
                                   'is_pc': is_pc_lst,
                                   'num_ubs': num_ubs_lst,
                                   })
    
    error_rates_df['strand'] = strand
    error_rates_df['type'] = "XNA"
    error_rates_df.loc[error_rates_df.is_pc, 'type'] = "PC"
    
    return error_rates_df

def pprint_agg_stats_error_rate(error_rates_df, max_dist=4, groupby_strand=True,
                                funcs=['mean', 'std'], format_only=False):
    new_label_index = ['no_ub','outside_ub_area','inside_ub_area','only_ub',]
    new_label_index += ['dist_ub_d-'+str(dist+1) for dist in range(max_dist)]

    if groupby_strand:
        summary_df = error_rates_df.groupby(['strand','type','label']).agg(
            {'error_rates': funcs})
        summary_df = summary_df.reindex(new_label_index, level=2).unstack(level=1)
    else:
        summary_df = error_rates_df.groupby(['type','label']).agg(
            {'error_rates': funcs})
        summary_df = summary_df.reindex(new_label_index, level=1).unstack(level=0)
    
    if not format_only:
        # print("Error rates:")
        with pd.option_context("display.float_format", '{:.1f}'.format):
            print(summary_df.T)
    
    return summary_df

def compute_sequencing_perf(paf_df, verbose=False, groupby='target_id',
                            summarized=True, 
                            funcs = ['mean', 'std'],
                            main_stat='mean', 
                            perf_cols=['target_cover','percent_match'],
                            summ_groupby_strand=True):
    perf_dict = dict(zip(perf_cols, len(perf_cols)*[funcs] ))
    
    results_by_tar_df = paf_df.groupby(['target_id','strand','type']).agg(perf_dict) * 100

    if summ_groupby_strand:
        results_df = paf_df.groupby(['strand','type']).agg(perf_dict) * 100
    else:
        results_df = paf_df.groupby(['type']).agg(perf_dict) * 100

    if verbose:
        with pd.option_context("display.float_format", '{:.2f}'.format):
            print("Outputting results groupped by 'target id' and 'strand'.")
            if verbose >= 2:
                print("\nStrand: Forward")
                print(results_by_tar_df.xs('F', level=1))
                print("\nStrand: Reverse")
                print(results_by_tar_df.xs('R', level=1))
            
            perf_dict = dict(zip( [ (c, 'mean') for c in perf_cols ], 
                                 len(perf_cols)*[funcs] ))
            print(results_by_tar_df.groupby(['strand','type',]).agg(
                perf_dict))
                # {('target_cover','mean'): funcs,
                #  ('percent_match','mean'): funcs}))
            
            print("Outputting results sorted by <metric> mean.")
            stats = ['median','mean','std']
            for metric in perf_cols:
                cols = [ (metric, stat) for stat in stats]
                print(results_by_tar_df[cols].sort_values((metric, 'mean')))
            
            print("\nOutputting results groupped only by strand.")
            print(results_df)
    
    if groupby == 'target_id':
        if not summarized:
            ret_df = results_by_tar_df
        else:
            perf_dict = dict(zip( [ (c, main_stat) for c in perf_cols ], 
                             len(perf_cols)*[[np.mean, np.std]] ))
            by_cols = ('type' if not summ_groupby_strand else ['strand','type'])
            
            # summary_df = results_by_tar_df.groupby(['strand','type',]).agg(
            # summary_df = results_by_tar_df.groupby('type').agg(
            summary_df = results_by_tar_df.groupby(by_cols).agg(
                # {('target_cover','mean'): funcs, ('percent_match','mean'): funcs})
                # {('target_cover',main_stat): [np.mean, np.std], 
                #  ('percent_match',main_stat): [np.mean, np.std]})
                perf_dict)
    
            # summary_df.columns = summary_df.columns.droplevel(1)
            
            ret_df = summary_df
    else:
        ret_df = results_df
    
    
    return ret_df

def compute_seq_perf_by_pos(paf_df, ref_info, exp, reads_dict=None):
    """
    Compute sequencing performace at base-level(target position), 
    with relevant labels related to base proximity to UB.

    Parameters
    ----------
    paf_df : pd.DataFrame
        DESCRIPTION.
    ref_info : TYPE
        DESCRIPTION.
    exp : str
        DESCRIPTION.
    reads_dict : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    error_rates_df : pd.DataFrame
        DataFrame with a row for each base position error rate plus repetition.
        ex:     target_id      label  error_rates  is_pc  num_ubs strand type
        0       XNA03        only_ub   100.000000  False        1      R  XNA
        1       XNA03          no_ub    12.285456  False        1      R  XNA
        2       XNA03          no_ub     9.852454  False        1      R  XNA
        ...
        649     XNA16  dist_ub_d-11+     1.006006   True        0      R   PC

    """
    #### [ub_area_*] compute_all_error_rates_paf appends info to paf_df 
    ### Func to compute error rate for each position, per target
    error_rates_dict = compute_all_error_rates_paf(paf_df, ref_info.targets,
                                                   reads_dict=reads_dict)
    
    error_rates_list = []
    for strand, paf_strand in zip(['F','R'],['+','-']):
        selected_targets = paf_df[paf_df.strand.isin([strand,paf_strand])]\
            .target_id.unique() # Compute only for those aligned
        # selected_targets = None # Compute for all targets
        
        ### Func to group error rates per position based on how close to UB
        error_rates_df = aggregate_stats_error_rate(exp, strand, None,
                                            selected_targets=selected_targets,
                                            error_rates_dict=error_rates_dict)
        error_rates_df['strand'] = strand
        error_rates_list.append(error_rates_df)

    error_rates_df = pd.concat(error_rates_list).reset_index(drop=True)
    
    return error_rates_df

def compute_analyze_seq_perf(paf_df, ref_info, exp_name, reads_filepath=None,
                             targets_breakdown=True,
                             verbose=False, groupby_strand=False, max_dist=4):
    print("Indexing reads file...")
    reads_dict = (None if reads_filepath is None else
                  SeqIO.index(reads_filepath, reads_filepath.split('.')[-1]))
    print("Number of reads on file: {:0,d} ({:.2%} aligned)".format(
        len(reads_dict), paf_df.shape[0]/len(reads_dict)))
        
    # print("Computing error rates per position...")
    error_rates_df = compute_seq_perf_by_pos(paf_df, ref_info, exp_name,
                                             reads_dict=reads_dict)
    
    perf_cols = ['percent_match']
    # perf_cols += ['target_cover', 'read_alignment_cover']
    if reads_dict is not None:
        perf_cols += ['target_acc','read_acc']
    
    summary_df = compute_sequencing_perf(paf_df, verbose=verbose,
                groupby=None, summarized=False,
                # funcs = [np.median, np.mean, np.std], 
                funcs = ['median', 'mean', 'std'], 
                perf_cols=perf_cols,
                summ_groupby_strand=groupby_strand
                )
    
    with pd.option_context("display.float_format", '{:.2f}'.format):
        # print("Outputting results groupped by 'target id' and/or 'strand'.")
        print("Results computed over all reads:")
        print(summary_df)
    
    #### Breakdown results per target + strand
    if targets_breakdown:
        print("Showing ub_area_acc_plus results mean grouped by target+strand:")
        mean_accs_df = paf_df.groupby(['target_id','strand']).ub_area_acc_plus.mean()
        with pd.option_context("display.float_format", '{:.1%}'.format):
            print(mean_accs_df.describe().to_frame().T.drop(columns=['count']))
            # print(mean_accs_df.sort_values().head(8).reset_index().T)
            print(mean_accs_df.sort_values().reset_index().T)
    
    # funcs = [np.mean, np.std]     
    err_summary_df = pprint_agg_stats_error_rate(error_rates_df,
                                max_dist=max_dist, 
                                # funcs=funcs,
                                groupby_strand=groupby_strand,
                                # format_only=True,
                               )
    
    #### Print summary line for copy and paste to spreadsheet
    if not groupby_strand:
        float_tpl = '{:.1f}'

        summary_info = {}
        # summary_info['num_aligned_reads'] = ('{:0,d}', len(paf_df.read_id.unique()) )
        summary_info['num_aligned_reads'] = ('{}', len(paf_df.read_id.unique()) )
        
        ### Version 0 stats and order
        # summary_info['acc_xna'] = (float_tpl, summary_df.loc['XNA',('percent_match','median','mean')] )
        # summary_info['acc_pc'] = (float_tpl, 
        #   np.nan if not has_pc else summary_df.loc['PC',('percent_match','median','mean')])
        
        if reads_dict is not None:
            summary_info['target_acc'] = (float_tpl, 
                   # summary_df.loc['XNA',('target_acc','median','mean')])
                    summary_df.loc['XNA',('target_acc','mean')])
            summary_info['read_acc'] = (float_tpl, 
                   # summary_df.loc['XNA',('read_acc','median','mean')])
                   summary_df.loc['XNA',('read_acc','median')])
                    
        err_column = ('error_rates','mean','XNA')
        summary_info['err_far_ub'] = (
            float_tpl, err_summary_df.loc['outside_ub_area',err_column] )
        summary_info['err_close_ub'] = (
            float_tpl, err_summary_df.loc['inside_ub_area',err_column] )
        summary_info['err_only_ub'] = (
            float_tpl, err_summary_df.loc['only_ub',err_column] )
        
        for dist in range(1, max_dist+1):
            summ_col = 'err_ub_d_{}'.format(dist)
            err_row = 'dist_ub_d-{}'.format(dist)
            summary_info[summ_col] = (
                float_tpl, err_summary_df.loc[err_row, err_column] )
        
        ### Version 0 stats and order
        # if reads_dict is not None:
        #     summary_info['target_acc'] = (float_tpl, 
        #             summary_df.loc['XNA',('target_acc','median','mean')])
        #     summary_info['read_acc'] = (float_tpl, 
        #             summary_df.loc['XNA',('read_acc','median','mean')])
            
        summary_info['acc_xna'] = (float_tpl, summary_df.loc['XNA',('percent_match','mean')] )
        
        has_pc = EXP_REF_MAP[exp_name] == 'XNA16' # Excluding XNA_4Ds also, because PC uses same BC
        summary_info['acc_pc'] = (float_tpl, 
          np.nan if not has_pc else summary_df.loc['PC',('percent_match','mean')])
    
        print("\nSpreadsheet summary info:")
        print(','.join(summary_info.keys()))
        print(','.join([ tpl.format(val) for (tpl,val) in summary_info.values()]))

def compute_read_confusion_matrix(read_info, ref_info, reads_filepath=None, reads_dict=None):
    read_seq = get_read_seq(read_info.read_id, reads_filepath=reads_filepath, 
                            read_info=read_info, reads_dict=reads_dict)
    target_matches = compute_read_matches(read_seq, read_info=read_info)
    
    target = ref_info.targets[read_info.target_id].replace('N','X')
    target_matches = polish_target_matches(target_matches, read_info, target)
    
    if read_info.strand in ['-','R']:
        target_matches = list(reverse_complement(target_matches))
        target = reverse_complement(target)
    target = list(target)
    
    cm_read = confusion_matrix(target, target_matches, labels=['A','T','C','G','X','Y','-'])
    cm_read = cm_read[:-1,:] # Removing '-' row

    return cm_read

def analyze_paf(exp_name, paf_filepath, min_reads_count=100, max_dist=0,
                targets_list_filepath=None, analyze_seq_perf=False, 
                reads_filepath=None, groupby_strand=False, include_list_filepath=None,
                count_tar_read=False, verbose=False, save_detailed_perf=False,
                max_bc_dist=None, save_res_summ=True, only_strand=None, ubs='XY',
                q_score_print=False, debug=False, save_perf_per_read=False,
                save_confusion_matrix=False, quiet=False):
    
    if exp_name not in VALID_REFS:
        ref_name = EXP_REF_MAP[exp_name]
    else:
        ref_name = exp_name
        exp_name = REF_EXP_MAP[ref_name][0]

    ref_info = XNA_refs(ref_name)
    
    # has_pc = ref_name != 'XNA1024'
    # has_pc = ref_name == 'XNA16' # Excluding XNA_4Ds also, because PC uses same BC
    
    print("Experiment arguments")
    print(f"> Selected sequecing run: {exp_name} ({ref_name})")
    # print("> Selected sequecing run:", exp_name)
    # print("> Reference template name:", ref_name)
    print("> paf_filepath:", paf_filepath)
    print("> reads_filepath:", reads_filepath)
    # print("> analyze_seq_perf:", analyze_seq_perf)
    if targets_list_filepath: print("> targets_list_filepath:", targets_list_filepath)
    if min_reads_count: print("> min_reads_count:", min_reads_count)
    if groupby_strand: print("> groupby_strand:", groupby_strand)
    
    #### Reading paf_df
    if '.paf' in paf_filepath[0]:
        print("Reading paf file(s)...")
        # paf_df = read_paf(paf_filepath)
        extra_tags = []
        if analyze_seq_perf:
            extra_tags.append('cs')
        paf_df = read_multiple_pafs(paf_filepath, extra_tags=extra_tags)
    else:
        print("Reading sam file...")
        paf_df = read_sam(paf_filepath[0])
    
    if debug:
        print("[Warning] Using subset of rows for debugging purposes...")
        # paf_df = paf_df[paf_df.read_id=='c4944bbf-2300-4d9f-9b81-462cfccaf8e8'].reset_index()
        # paf_df = paf_df[(paf_df.target_id=='TTCCT') & (paf_df.strand=='-')]
        paf_df = paf_df.head(1000).copy()
    
    # print("*** paf number of alignments: {:0,d}".format(paf_df.shape[0]))
    align_cnt = paf_df.read_id.nunique()
    # print(f"* number of unique read ids: {align_cnt:0,d}")
    print(f"* paf contains {align_cnt:0,d} reads ({len(paf_df):0,d} alignments)")
    
    if reads_filepath is not None:
        print("Indexing reads file...")
        reads_dict = SeqIO.index(reads_filepath, reads_filepath.split('.')[-1])
        print("* Number of reads on file: {:0,d} ({:.2%} aligned)".format(
            len(reads_dict), paf_df.read_id.nunique()/len(reads_dict)))
            # len(reads_dict), paf_df.shape[0]/len(reads_dict)))
        reads_codename = os.path.splitext(os.path.basename(reads_filepath))[0]
        if reads_codename.startswith('reads-'):
            reads_codename = reads_codename[6:]
    
    # out_prefix = f'results_summ-{exp_name}-{reads_codename}'
    out_prefix = f'results_summ-{reads_codename}'
    out_dir = os.path.dirname(paf_filepath[0])
    
    if targets_list_filepath is not None:
        print("Filtering paf by input targets list...")
        targets_list = pd.read_csv(targets_list_filepath, header=None, squeeze=True)
        
        print(f"> {len(targets_list)=}")
        paf_df = paf_df[paf_df.target_id.isin(targets_list)].reset_index(drop=True)
        print("* paf number of reads with selected targets: {:0,d}".format(paf_df.shape[0]))
    
    if include_list_filepath is not None:
        ids_list = read_tsv(include_list_filepath).read_id
        count_matches = paf_df.read_id.isin(ids_list).sum()
        paf_df = paf_df[paf_df.read_id.isin(ids_list)]
        print("Keeping reads present in include list:",
              f"{count_matches:0,d} out of {len(ids_list):0,d} on list.")
        # print(f"remaining number of reads: {len(paf_df):0,d}")
    
    # return paf_df ### Before adding barcode info
    
    #### Filter by barcode distance [max_bc_dist] 
    if max_bc_dist is not None:
        print("Adding barcode information...")
        add_barcode_info(paf_df, ref_info, reads_dict, inplace=True, parallel=True)
        # paf_df = add_barcode_info(paf_df, ref_info, reads_dict, parallel=True)
        paf_df = paf_df[paf_df.barcode_distance <= max_bc_dist].reset_index(drop=True)
        
        ### Keep min bc for repeated read_ids
        paf_df = paf_df[paf_df.barcode_distance == 
                        paf_df.groupby('read_id').barcode_distance.transform('min')]
        # dup_paf_df = paf_df[paf_df.read_id.duplicated(keep=False)]
        # if len(dup_paf_df) > 0:
        #     dup_read_cnt = dup_paf_df.read_id.nunique()
        #     print(f"[WARNING] Duplicated {dup_read_cnt} read_id aligns with same barcode distance:")
        #     print(dup_paf_df[['target_id','strand','barcode_distance','read_id']])
        
        # demux_cnt = len(paf_df)
        demux_cnt = paf_df.read_id.nunique()
        print("Filtering by barcode distance, max:", max_bc_dist)
        print("* Remaining number of unique read ids: {:0,d} ({:0.1%})".format(
            demux_cnt, demux_cnt/len(reads_dict) ))
    
    if paf_df.empty:
        print("No read left to analyze performance, exiting...")
        return
        # print("Read satisfy the ")
    
    paf_df.strand.replace({'+':'F', '-':'R'}, inplace=True)
    
    if ubs != 'XY':
        only_strand = dict(X='F', Y='R')[ubs]
        
    if only_strand is not None:
        only_strand = only_strand.replace('+','F').replace('-','R')
        print(f"[WARNING] Filtering reads, keeping only strand '{only_strand}'")
        paf_df = paf_df[paf_df.strand==only_strand].reset_index(drop=True)
    
    has_pc = any(paf_df.is_pc)
    # print("Alignment file contains PC:", has_pc)
    
    if verbose and not analyze_seq_perf:
        print_reads_stats(paf_df, ref_info)
    
    #### Quality Score
    if q_score_print:
        print("Computing mean quality scores for mapped region...")
        paf_df['mean_q_score'] = paf_df.progress_apply(lambda row:
            np.mean(get_read_qual(row.read_id, None, read_info=row, reads_dict=reads_dict)),
            axis='columns')
        print()
            
        with pd.option_context("display.float_format", '{:.1f}'.format):
            # print(paf_df[['mean_q_score']].describe([.01,.05,.10,.25]).T)
            print(paf_df.groupby(['strand']).mean_q_score.describe(
                percentiles=[.01,.05,.10,.25]))
            if paf_df.target_id.nunique() <= 20:
                print(paf_df.groupby(['target_id','strand']).mean_q_score.describe(
                    percentiles=[.01,.05,.10,.25]))
    
    # if not analyze_seq_perf:
    if count_tar_read:
        #### Computing target read count
        min_tar_reads_count_df = get_tar_reads_count(paf_df, ref_info.targets_id,
                                                     print_stats=True)
        
        if targets_list_filepath is not None:
            # print(min_tar_reads_count_df.n_read.describe())
            # print(min_tar_reads_count_df.groupby('n_read').size())
            print(min_tar_reads_count_df[min_tar_reads_count_df.target_id.isin(targets_list)]
                  .groupby('n_read').size().to_frame().T)
        
        if min_reads_count is not None:
            missing_mask = (min_tar_reads_count_df.n_read <= min_reads_count)
            missing_targets = min_tar_reads_count_df[missing_mask].target_id
            print("Number of missing targets (F and/or R):", len(missing_targets))
            # print(missing_targets.values)
        
        if False:
            paf_dir, paf_file = os.path.split(paf_filepath)
        
            missing_file = 'miss_tpl-r_{}-'.format(min_reads_count) + paf_file + '.txt'
            missing_filepath = os.path.join(paf_dir, missing_file)
            missing_targets.to_csv(missing_filepath, header=False, index=False)
            print("Generated missing targets file:", missing_filepath)
    
    # %% analyze_seq_perf
    if analyze_seq_perf:
        # print("Computing error rates per position...")
        error_rates_df = compute_seq_perf_by_pos(paf_df, ref_info, exp_name,
                                                 reads_dict=reads_dict)
        
        cnt_dups = paf_df.duplicated('read_id').sum()
        if cnt_dups > 0:
            # print(f"[WARNING] Dropping duplicated read_ids ({cnt_dups}),", 
            #       "keep max ub_area_acc_plus. Re-computing error rates...")
            paf_df = paf_df.sort_values('ub_area_acc_plus').groupby('read_id').tail(1)\
                .reset_index(drop=True)
            # print("Re-computing error rates per position...")
            error_rates_df = compute_seq_perf_by_pos(paf_df, ref_info, exp_name,
                                                     reads_dict=reads_dict)

        #### Last modification to paf_df
        # return paf_df # AFTER Dropping duplicates!
        
        #### Save confusion matrix over full-length aligned region
        if save_confusion_matrix:
            print("Computing Confusion Matrix (full-length aligned read)...")
            cm_reads = paf_df.progress_apply(compute_read_confusion_matrix, 
            # cm_reads = paf_df.apply(compute_read_confusion_matrix, 
                axis=1, ref_info=ref_info, reads_dict=reads_dict)
            cm = np.sum(cm_reads, axis=0)
            
            cm_filepath = os.path.join(out_dir, out_prefix+'-confusion_matrix.npy')
            print("Saving file:", os.path.basename(cm_filepath))
            np.save(cm_filepath, cm)
        
        #### Save performance per read
        if save_perf_per_read:
            out_filepath = os.path.join(out_dir, out_prefix+'-by_read.csv.gz')
            print("Saving file:", os.path.basename(out_filepath))
            drop_cols = ['read_length', 'read_start', 'read_end', 'target_length', 
                         'target_start', 'target_end', 'n_matches', 'block_length', 
                         'mapping_quality', 'cs']
            perf_per_read_df = paf_df.drop(columns=drop_cols, errors='ignore')
            perf_per_read_df = perf_per_read_df.round(6)
            perf_per_read_df.to_csv(out_filepath, header=True, index=False)
        
        perf_cols = ['percent_match']
        # perf_cols += ['target_cover', 'read_alignment_cover']
        if reads_dict is not None:
            perf_cols += ['target_acc','read_acc']
            perf_cols += ['ub_acc','ub_area_acc','non_ub_area_acc']
            perf_cols += ['fpr']
        
        #### Results detailed by target+strand
        if save_detailed_perf:
            results_by_tar_df = compute_sequencing_perf(paf_df, verbose=verbose,
                        groupby='target_id', summarized=False,
                        # funcs = [np.median, np.mean, np.std],
                        funcs = 'mean',
                        perf_cols=perf_cols+['ub_area_acc_plus'],
                        # summ_groupby_strand=groupby_strand,
                        summ_groupby_strand=False,
                        )
            # stats = ['mean','std']
            # for metric in perf_cols+['ub_area_acc_plus']:
            #     cols = [ (metric, stat) for stat in stats]
            #     print(results_by_tar_df[cols].sort_values((metric, 'mean')))
            
            # return results_by_tar_df
            
            read_cnt_df = paf_df.groupby(['target_id','strand','type']).agg({'read_id':'size'})
            results_by_tar_df = results_by_tar_df.merge(read_cnt_df, 
                                            left_index=True, right_index=True)
            
            #### TODO Add row for target+str with 0 reads
            # 1) create IDX only df, 2) append to results_by_tar_df, 3) replace nan with 0
            # all_target_read_count_df = pd.DataFrame(
            #     {'target_id': 2*targets_id, 
            #      'strand': len(targets_id)*['F'] + len(targets_id)*['R'], 
            #      'n_read': 2*len(targets_id)*[0]})
            
            if not quiet:
                print("Statistics of results detailed by target and strand:")
                # print_cols = ['ub_acc','ub_area_acc','ub_area_acc_plus','non_ub_area_acc','read_id']
                print_cols = ['ub_acc','ub_area_acc','non_ub_area_acc','read_id']
                if paf_df.target_id.nunique() > 20:
                    with pd.option_context("display.float_format", '{:.1f}'.format):
                        print(results_by_tar_df[print_cols].describe(
                                [.01,.05,.10,.25,.75,.90,.95,.99]))
                else:
                    with pd.option_context("display.float_format", '{:.1f}'.format):
                        print(results_by_tar_df[print_cols].unstack(1)
                            .swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
                            .droplevel('type')
                            .rename(columns={'ub_acc': 'ub', 'ub_area_acc': 'ub_area', 
                                             'ub_area_acc_plus': 'ub_area_+',
                                             'non_ub_area_acc': 'non_ub',})
                            )
                
                with pd.option_context("display.float_format", '{:.1f}'.format):
                    print(results_by_tar_df.reset_index().groupby('strand')[print_cols].mean()
                          .rename(columns={'ub_acc': 'ub', 'ub_area_acc': 'ub_area', 
                                           'ub_area_acc_plus': 'ub_area_+',
                                           'non_ub_area_acc': 'non_ub',})
                          .unstack(0).to_frame().T.swaplevel(axis=1)
                          .sort_index(axis=1, level=0, sort_remaining=False)
                          .rename(index={0:'mean     '}))
            
            if save_res_summ:
                out_filepath = os.path.join(out_dir, out_prefix+'-by_tar.csv')
                print("Saving file:", os.path.basename(out_filepath))
                results_by_tar_df.to_csv(out_filepath, 
                            # sep='\t', 
                            na_rep='nan',
                            header=True, index=True, float_format='{:.3f}'.format)
            
            #### Results detailed by ub position
            # if paf_df.target_id.nunique() == 4: # For XNA 20 only
            if paf_df.label_per_pos.apply(len).max() > 1: # For XNA 20 only
                func_dict = {}
                first_cols = ['is_pc', 'type', 'label_per_pos', 'barcode',]
                func_dict.update({ col:'first' for col in first_cols })
                per_pos_cols = ['ub_acc_per_pos', 'ub_area_acc_per_pos', 'ub_area_acc_plus_per_pos',]
                mean_lists = lambda lists: np.mean([ l for l in lists ], axis=0, keepdims=True).tolist()[0]
                func_dict.update({ col:mean_lists for col in per_pos_cols })
                
                # paf_df.target_id = paf_df.target_id.replace(ref_info.aliases)
                # agg_paf_df = paf_df.groupby(['strand','target_id']).agg(func_dict)#.reset_index()
                agg_paf_df = (paf_df[~paf_df.is_pc].
                # agg_paf_df = (paf_df[(~paf_df.is_pc) & (paf_df.label_per_pos.apply(len)>1)].
                              groupby(['strand','target_id']).agg(func_dict))
                
                # print("Exploding results per position:")
                per_pos_cols = ['label_per_pos','ub_acc_per_pos', 'ub_area_acc_per_pos', 'ub_area_acc_plus_per_pos']
                from packaging import version
                if version.parse(pd.__version__) >= version.parse("1.3.0"):
                    exp_paf_df = agg_paf_df[per_pos_cols].explode(per_pos_cols).rename(columns=(lambda c: c[:-8])).reset_index()
                else:
                    exp_paf_dfs = []
                    for col in per_pos_cols:
                        exp_paf_dfs.append(agg_paf_df[[col]].explode(col).transform(pd.to_numeric))
                    exp_paf_df = pd.concat(exp_paf_dfs, axis=1).rename(columns=(lambda c: c[:-8])).reset_index()
                
                x_pos = paf_df.groupby(['strand','target_id']).label_per_pos.first().to_dict()
                exp_paf_df['ub_order'] = exp_paf_df.apply(
                    lambda row: x_pos[(row.strand,row.target_id)].index(row.label)+1, axis=1)
                
                if not quiet:
                    with pd.option_context("display.float_format", '{:.1%}'.format):
                        print(exp_paf_df[
                            ['ub_acc','ub_area_acc','ub_area_acc_plus','strand','target_id','ub_order']]
                            .set_index(['strand','target_id','ub_order'])
                            .unstack(0)
                            .swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
                            .rename(columns={'ub_acc': 'ub', 'ub_area_acc': 'ub_area', 
                                             'ub_area_acc_plus': 'ub_area_plus'}))
                    
                    ### Print Means per pos as well (same as err_summary_df)
                    with pd.option_context("display.float_format", '{:.1%}'.format):
                        print(exp_paf_df.groupby('strand')[['ub_acc','ub_area_acc','ub_area_acc_plus']]
                              .agg(['mean','median'])
                              .rename(columns={'ub_acc': 'ub', 'ub_area_acc': 'ub_area',
                                               'ub_area_acc_plus':'ub_area_plus'})
                              .stack(1).unstack(0).swaplevel(axis=1)
                              .sort_index(axis=1, level=0, sort_remaining=False)
                              .rename(index={'mean':'          mean    ','median':'          median  '})
                             )
                
                if save_res_summ:
                    out_filepath = os.path.join(out_dir, out_prefix+'-by_tar_pos.csv')
                    print("Saving file:", os.path.basename(out_filepath))
                    exp_paf_df[['ub_acc','ub_area_acc','ub_area_acc_plus']] *= 100
                    exp_paf_df.to_csv(out_filepath, 
                                # sep='\t', 
                                na_rep='nan',
                                header=True, index=False, float_format='{:.3f}'.format)
        
        ### Computes performance over all reads. No group by target_id.
        summary_df = compute_sequencing_perf(paf_df, verbose=verbose,
                    groupby=None, summarized=False,
                    funcs = ['mean', 'std', 'median'], 
                    perf_cols=perf_cols, summ_groupby_strand=groupby_strand)
        
        
        print_ungrouped_results = False
        if print_ungrouped_results:
            with pd.option_context("display.float_format", '{:.2f}'.format):
                # print("Results groupped by 'target id' and/or 'strand'.")
                print("Results computed over all reads:")
                # print(paf_df[['ub_acc','ub_area_acc','ub_area_acc_plus',
                #               'non_ub_area_acc','read_id']].describe().T)
                # print(summary_df.T.unstack(1))
                print(summary_df.T.unstack(1).T)
                # print(summary_df[perf_cols[:3]])
                # print(summary_df[perf_cols[3:]])
        
        err_summary_df = pprint_agg_stats_error_rate(
            error_rates_df, max_dist=max_dist, groupby_strand=groupby_strand,
            format_only=True,) # funcs=[np.mean, np.std],
        
        if not groupby_strand:
            float_tpl = '{:.1f}'
    
            summary_info = {}
            # summary_info['num_aligned_reads'] = ('{:0,d}', len(paf_df.read_id.unique()) )
            summary_info['num_aligned_reads'] = ('{}', len(paf_df.read_id.unique()) )
            
            ### Version 0 stats and order
            # summary_info['acc_xna'] = (float_tpl, summary_df.loc['XNA',('percent_match','median','mean')] )
            # summary_info['acc_pc'] = (float_tpl, 
            #   np.nan if not has_pc else summary_df.loc['PC',('percent_match','median','mean')])
            
            if reads_dict is not None:
                summary_info['target_acc'] = (float_tpl, 
                       # summary_df.loc['XNA',('target_acc','median','mean')])
                        summary_df.loc['XNA',('target_acc','mean')])
                summary_info['read_acc'] = (float_tpl, 
                       # summary_df.loc['XNA',('read_acc','median','mean')])
                       summary_df.loc['XNA',('read_acc','mean')])
                        
            err_column = ('error_rates','mean','XNA')
            summary_info['err_far_ub'] = (
                float_tpl, err_summary_df.loc['outside_ub_area',err_column] )
            summary_info['err_close_ub'] = (
                float_tpl, err_summary_df.loc['inside_ub_area',err_column] )
            summary_info['err_only_ub'] = (
                float_tpl, err_summary_df.loc['only_ub',err_column] )
            
            for dist in range(1, max_dist+1):
                summ_col = 'err_ub_d_{}'.format(dist)
                err_row = 'dist_ub_d-{}'.format(dist)
                summary_info[summ_col] = (
                    float_tpl, err_summary_df.loc[err_row, err_column] )
            
            ### Version 0 stats and order
            # if reads_dict is not None:
            #     summary_info['target_acc'] = (float_tpl, 
            #             summary_df.loc['XNA',('target_acc','median','mean')])
            #     summary_info['read_acc'] = (float_tpl, 
            #             summary_df.loc['XNA',('read_acc','median','mean')])
                
            summary_info['acc_xna'] = (float_tpl, summary_df.loc['XNA',('percent_match','mean')] )
            summary_info['acc_pc'] = (float_tpl, 
              np.nan if not has_pc else summary_df.loc['PC',('percent_match','mean')])
        
            
            # res_df['ub_acc'] = 100 - res_df.err_only_ub
            # res_df['ub_area_acc'] = 100 - res_df.err_close_ub
            # res_df['non_ub_area_acc'] = 100 - res_df.err_far_ub
            
            summ_df = pd.DataFrame({key:[val] for key,(tpl,val) in summary_info.items()})
            pretty_summ_df = 100-summ_df[['err_only_ub','err_close_ub','err_far_ub']]
            # pretty_summ_df.columns = ['ub_acc', 'ub_area_acc', 'non_ub_area_acc']
            pretty_summ_df.columns = ['ub', 'ub_A', '~ub_A']
            
            ### Computing across all reads (different than err_summary_df)
            # pretty_summ_df['ub_acc_paf'] = [100*paf_df.ub_matches.sum() / paf_df.ub_len.sum()]
            # pretty_summ_df['ub_area_acc_paf'] = [100*paf_df.ub_area_matches.sum() / paf_df.ub_area_len.sum()]
            # pretty_summ_df['non_ub_area_acc_paf'] = [
            #     100*paf_df.non_ub_area_matches.sum() / paf_df.non_ub_area_len.sum()]
            # pretty_summ_df['ub_area_acc_plus_R'] = 100*( (paf_df.ub_area_matches.sum() + paf_df.ub_matches.sum())
            #                       /(paf_df.ub_area_len.sum()+paf_df.ub_len.sum()) )
            
            ### Computing first across all reads within same S+T, then averaging between them
            # # pretty_summ_df['ub_acc_paf'] = (100*paf_df.groupby(['strand','target_id']).ub_matches.sum() / paf_df.groupby(['strand','target_id']).ub_len.sum()).mean()
            # # pretty_summ_df['ub_area_acc_paf'] = (100*paf_df.groupby(['strand','target_id']).ub_area_matches.sum() / paf_df.groupby(['strand','target_id']).ub_area_len.sum()).mean()
            # # pretty_summ_df['non_ub_area_acc_paf'] = (
            # #     100*paf_df.groupby(['strand','target_id']).non_ub_area_matches.sum() / paf_df.groupby(['strand','target_id']).non_ub_area_len.sum()).mean()
            # pretty_summ_df['ub_A+'] = ( # ub_area_plus_acc
            #     100*(paf_df.groupby(['strand','target_id']).ub_area_matches.sum() + paf_df.groupby(['strand','target_id']).ub_matches.sum())
            #     /(paf_df.groupby(['strand','target_id']).ub_area_len.sum()+paf_df.groupby(['strand','target_id']).ub_len.sum()) ).mean()
            # Similar/same to: 100*paf_df.groupby(['strand','target_id']).ub_area_acc_plus.mean().mean()
            
            # pretty_summ_df['demux'] = 100*len(paf_df)/len(reads_dict)
            pretty_summ_df['demux'] = 100*demux_cnt/len(reads_dict)
            pretty_summ_df['align'] = 100*align_cnt/len(reads_dict)
            summary_info['demux'] = (float_tpl, pretty_summ_df['demux'].squeeze())
            summary_info['align'] = (float_tpl, pretty_summ_df['align'].squeeze())
            summ_df['demux'] = pretty_summ_df['demux']
            summ_df['align'] = pretty_summ_df['align']
            
            # pretty_summ_df = pretty_summ_df.reindex(sorted(pretty_summ_df.columns), axis=1)
            #### Print summary lines to copy+paste in spreadsheet
            print("\nPretty print summary info (Means over ub_pos+target_id+strand):")
            with pd.option_context("display.float_format", '{:.1f}'.format):
                print(pretty_summ_df)
            
            #### false_positive_rate + specificity
            mean_fpr = paf_df.fpr.mean()
            print(f"Mean False Positive Rate: {mean_fpr:.0%} ({1-mean_fpr:.1%} specificity)")
            # FPR = FP/(FP+TN) # specificity = 1 - FPR
            summ_df['specificity'] = 100*(1-mean_fpr) # mean_specificity
            mean_fdr = paf_df.fdr.mean()
            print(f"Mean False Discovery Rate: {mean_fdr:.0%} ({1-mean_fdr:.0%} precision)")
            # FDR = FP/(FP+TP) # precision = 1 - FDR
            summ_df['precision'] = 100*(1-mean_fdr) # mean_precision
            
            agg_outs = paf_df[['true_pos','false_neg','false_pos','true_neg']].sum(axis=0)
            tp, fn, fp, tn = agg_outs
            recall = tp/(tp+fn) if tp+fn > 0 else 0
            precision = tp/(tp+fp) if tp+fp > 0 else 0
            f1_score = 2*tp/(2*tp+fp+fn) if tp+fp+fn > 0 else 0
            specificity = tn/(fp+tn) if fp+tn > 0 else 0
            print(f"F₁-Score: {f1_score:.1%} ({recall=:.1%} | {precision=:.1%})")
            # summ_df['specificity'] = 100*specificity
            # summ_df['recall'] = 100*recall
            # summ_df['precision'] = 100*precision
            summ_df['f1_score'] = 100*f1_score
            
            beta = 2 # heavier weight to recall
            if precision+recall > 0:
                fbeta_score = (1+beta**2)*precision*recall/(beta**2*precision+recall)
            else:
                fbeta_score = 0
            summ_df[f'f{beta}_score'] = 100*fbeta_score
            print(f"F{beta}-Score: {fbeta_score:.1%}")
            
            for k,v in zip(['true_pos','false_neg','false_pos','true_neg'],[tp, fn, fp, tn]):
                summ_df[k] = int(v)
        
            # print("\nSpreadsheet summary info:")
            # print(','.join(summary_info.keys()))
            # print(','.join([ tpl.format(val) for (tpl,val) in summary_info.values()]))
            
            #### Saving Results Summary
            if save_res_summ:
                out_filepath = os.path.join(out_dir, out_prefix+'.csv')
                print("Saving file:", os.path.basename(out_filepath))
                summ_df.to_csv(out_filepath, na_rep='nan', header=True, 
                               index=False, float_format='{:.3f}'.format)
            
            return summ_df
    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting analyze_paf - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    analyze_paf(**args)

    print('\n> Finished analyze_paf -', time.asctime( time.localtime(time.time()) ))
