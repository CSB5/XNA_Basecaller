#!/usr/bin/env python
import argparse, sys, os, time
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(sys.path[0]))
from misc.xna_refs import XNA_refs

TEST_ALIASES = {'poc': 'POC-test', 'cplx': 'CPLX-test'}

def load_args():
    ap = argparse.ArgumentParser(
        description='Compare basecalling performances from 1+ trainings.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('basecalls_summ_filepaths',
        help='path to the basecalls_summ_filepath.',
        nargs='+', type=str)
    
    ap.add_argument('-t', '--test_label',
        help=(f"label of the test summary file to use (e.g. {'|'.join(list(TEST_ALIASES.keys()))})."
              " Use commas to report results from multiple test sets."),
        default='poc,cplx', type=str)
    ap.add_argument('-d', '--detailed_print',
        help='print results detailed per pos + target + strand.',
        action='store_true')
    ap.add_argument('-s', '--sort_ub_acc',
        help='sort by UB acc. and keep top 20',
        action='store_true')
    ap.add_argument('-a', '--detail_agg',
        help='how to aggregate results per pos + target + strand.',
        default='mean')
    ap.add_argument('-w', '--weights',
        help='print performance from particular weights.',
        type=int)
    
    ap.add_argument('-D', '--dict-print',
        help='plot main metrics from input train dirs.',
        action='store_true')
    ap.add_argument('-c', '--csv_print',
        help='print csv lines at the end, to copy to spreadsheet.',
        action='store_true')
    ap.add_argument('-o', '--old_style',
        help='print csv columns with old style.',
        action='store_true')

    # Optional arguments
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def comp_basecalls_perf(test_label, basecalls_summ_filepaths, csv_print=False,
                        detailed_print=False, old_style=False, sort_ub_acc=False,
                        dict_print=False, weights=None, detail_agg='mean',
                        print_params=True):
    basecalls_dir = 'basecalls' if not weights else f'basecalls-weights_{weights}'
    if print_params:
        print(f"> {basecalls_dir=}")
        if weights: print(f"> {weights=}")
        if detailed_print: print(f"> {detailed_print=} | {detail_agg=}")
        if csv_print: print(f"> {csv_print=} | {old_style=}")
        print()
    
    if ',' in test_label:
        test_labels = test_label.split(',')
        for test_label in test_labels:
            comp_basecalls_perf(test_label, basecalls_summ_filepaths, csv_print=csv_print,
                detailed_print=detailed_print, old_style=old_style, sort_ub_acc=sort_ub_acc,
                dict_print=dict_print, weights=weights, detail_agg=detail_agg, 
                print_params=False)
        return
    test_label = TEST_ALIASES.get(test_label, test_label)
    ### Note: [DEPREC] test_label=20 is a special case that manually combines 16+4ds
    
    print(f"> Results file: results_summ-{test_label}.csv")
    
    #### Reading summary files
    basecalls_summ_dfs = []
    for basecall_dir in basecalls_summ_filepaths:
        if basecall_dir.endswith('.csv'):
            basecalls_summ_filepath = basecall_dir
        else:
            basecalls_summ_filepath = os.path.join(basecall_dir, basecalls_dir,
                                                   f"results_summ-{test_label}.csv")
            if not os.path.exists(basecalls_summ_filepath):
                basecalls_summ_filepath = os.path.join(basecall_dir,
                                                   f"results_summ-{test_label}.csv")
                
        if not os.path.exists(basecalls_summ_filepath) and test_label != '20':
            # print(f'[WARNING]: Results file not found, skipping dir: {basecall_dir}')
            continue
        
        if test_label != '20':
            basecalls_summ_df = pd.read_csv(basecalls_summ_filepath)
        else:
            #### Merge 16 + 4ds - test_label = 20
            summ_df, summ_by_tar_pos_df = [], []
            test_aliases = ['A003-test_A003','A007-10k_filt']
            for test_alias in test_aliases:
                summ_by_tar_pos_filepath = os.path.join(basecall_dir,basecalls_dir, 
                    f'results_summ-{test_alias}-by_tar_pos.csv')
                if not os.path.exists(summ_by_tar_pos_filepath):
                    break
                temp_df = pd.read_csv(summ_by_tar_pos_filepath)
                num_tars_strands = temp_df.groupby(['strand','target_id']).ngroups
                summ_by_tar_pos_df.append(temp_df)
                
                summ_filepath = os.path.join(basecall_dir,basecalls_dir,
                    f'results_summ-{test_alias}.csv')
                if not os.path.exists(summ_filepath):
                    break
                temp_df = pd.read_csv(summ_filepath)
                temp_df['num_tars_strands'] = num_tars_strands
                summ_df.append(temp_df)
            if summ_df == []:
                # print(f'[WARNING]: Results file not found, skipping dir: {basecall_dir}')
                continue
            summ_df = pd.concat(summ_df).reset_index(drop=True)
            summ_by_tar_pos_df = pd.concat(summ_by_tar_pos_df).reset_index(drop=True)
            
            summ_by_tar_pos_df.target_id.replace(XNA_refs('XNA_4Ds').aliases, inplace=True)
            
            # cols = ['err_far_ub', 'demux', 'align', 'specificity', 'precision', 'f1_score']
            cols = ['err_far_ub', 'demux', 'align',]
            extra_cols = ['specificity', 'precision', 'f1_score','f2_score']
            for col in extra_cols:
                if col in summ_df:
                    cols.append(col)
            agg_summ_df = pd.concat((summ_by_tar_pos_df[['ub_acc','ub_area_acc']].mean(), 
                summ_df[cols].apply(lambda x: np.average(x, weights=summ_df.num_tars_strands))))
            agg_summ_df['num_aligned_reads'] = summ_df.num_aligned_reads.sum()
            agg_summ_df['err_only_ub'] = 100-agg_summ_df.ub_acc
            agg_summ_df['err_close_ub'] = 100-agg_summ_df.ub_area_acc
            
            basecalls_summ_df = agg_summ_df.to_frame().T

        basecalls_summ_df['exp'] = basecall_dir
        
        #### Detailed reading and printing (target+strand)
        if detailed_print:
            if test_label != '20':
                detailed_summ_filepath = os.path.join(
                    os.path.dirname(basecalls_summ_filepath),
                    # f"results_summ-{test_label}-by_tar.csv")
                    f"results_summ-{test_label}-by_tar_pos.csv")
                if os.path.exists(detailed_summ_filepath):
                    results_by_tar_df = pd.read_csv(detailed_summ_filepath)
                else:
                    detailed_summ_filepath = os.path.join(
                        os.path.dirname(basecalls_summ_filepath),
                        f"results_summ-{test_label}-by_tar.csv")
                    results_by_tar_df = pd.read_csv(detailed_summ_filepath)
                    results_by_tar_df['ub_order'] = 1
                    
                results_by_tar_df.target_id.replace(XNA_refs('XNA_4Ds').aliases, inplace=True)
            else:
                results_by_tar_df = summ_by_tar_pos_df
            results_by_tar_df.target_id.replace(XNA_refs('XNA_4Ds').aliases, inplace=True)
            
            results_by_tar_df = results_by_tar_df.set_index(['target_id','strand','ub_order'])
            
            
            if not csv_print:
                # print_cols = [...,'ub_area_acc_plus','non_ub_area_acc','read_id']
                print_cols = ['ub_acc','ub_area_acc']
                print("*****", basecall_dir, 10*"*")
                if len(results_by_tar_df) > 72:
                    with pd.option_context("display.float_format", '{:.1f}'.format):
                        percs = [0.01,0.05,.10,.25,.75]
                        # print(results_by_tar_df[print_cols].describe(percs).T)
                        print(results_by_tar_df[['ub_acc']].describe(percs).T)
                        
                    grp_tar_res_df = (results_by_tar_df.ub_acc.unstack('strand'))
                    grp_tar_res_df.reset_index(level='ub_order', drop=True, inplace=True)
                    grp_tar_res_df['min'] = grp_tar_res_df.min(axis=1)
                    strands = results_by_tar_df.index.get_level_values('strand').unique().values
                    grp_tar_res_df['sum'] = grp_tar_res_df[strands].sum(axis=1)
                    grp_tar_res_df = grp_tar_res_df.sort_values(['min','sum']).drop(columns=['min','sum'])
                    top_k = 14
                    print(f"Top {top_k} lowest UB Acc.:")
                    print(grp_tar_res_df.head(top_k).T.rename_axis(None, axis=1)
                          .rename_axis(None).round(1))
                else:
                    with pd.option_context("display.float_format", '{:.0f}'.format):
                        print(results_by_tar_df[print_cols].unstack(1)
                            .swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
                            .rename(columns={'ub_acc': 'ub', 'ub_area_acc': 'ub_area', 
                                             'ub_area_acc_plus': 'ub_area_+',
                                             'non_ub_area_acc': 'non_ub',})
                            )
                
                with pd.option_context("display.float_format", '{:.1f}'.format):
                    print(results_by_tar_df.reset_index().groupby('strand')[print_cols]
                          .agg(['mean']) # .agg(['mean','median'])
                          .rename(columns={'ub_acc': 'ub', 'ub_area_acc': 'ub_area', 
                                           'ub_area_acc_plus': 'ub_area_+',
                                           'non_ub_area_acc': 'non_ub',}))
            
            for ub, strand in [('X','F'),('Y','R')]:
                if strand in results_by_tar_df.index.get_level_values('strand'):
                    basecalls_summ_df[f'ub_{ub}'] = results_by_tar_df.xs(
                        strand, level="strand").ub_acc.agg(detail_agg)
                basecalls_summ_df['err_only_ub'] = 100 - results_by_tar_df.ub_acc.agg(detail_agg)
        basecalls_summ_dfs.append(basecalls_summ_df)
    
    if len(basecalls_summ_dfs) == 0:
        print('[WARNING]: No files found, exiting...')
        return
    summ_df = pd.concat(basecalls_summ_dfs).sort_index()
    
    if sort_ub_acc:
        print('[WARNING]: Sorting by UB Acc...')
        print('[WARNING]: Keeping top only...')
        summ_df = summ_df.sort_values('err_only_ub').head(20).reset_index(drop=True)
    
    #### Preparing pretty print dataframe
    pretty_summ_df = 100-summ_df[['err_only_ub','err_close_ub','err_far_ub']]
    # pretty_summ_df.columns = ['ub_acc', 'ub_area_acc', 'non_ub_area_acc']
    pretty_summ_df.columns = ['ub', 'ub_A', '~ub_A']
    
    extra_fields = ['demux','align','specificity','precision','f1_score','f2_score']
    for field in extra_fields:
        if field in summ_df:
            pretty_summ_df[field] = summ_df[field]
    pretty_summ_df.rename(columns={'specificity':'spec.','precision':'prec.',
                                   'f1_score':'F₁', 'f2_score':'F₂',
                                   }, 
                          inplace=True)
    
    if 'num_aligned_reads' not in pretty_summ_df:
        pretty_summ_df['num_align'] = summ_df['num_aligned_reads']
        
    if detailed_print:
        cols_order = pretty_summ_df.columns.tolist()
        extra_fields = ['ub_Y','ub_X']
        for field in extra_fields:
            if field in summ_df:
                pretty_summ_df[field] = summ_df[field]
                cols_order = [field] + cols_order
        pretty_summ_df = pretty_summ_df[cols_order]
    
    if csv_print:
        # if pretty_summ_df.ub.max() == 0:
        #     print("[WARNING] Removing UB accs cols because max is zero.")
        #     pretty_summ_df.drop(columns=['ub', 'ub_A'], inplace=True)
        
        if not old_style:
            pretty_summ_df.drop(columns='num_align', inplace=True)
            pretty_summ_df.drop(columns='spec.', inplace=True, errors='ignore')
            pretty_summ_df.drop(columns='prec.', inplace=True, errors='ignore')
            pretty_summ_df.drop(columns='F₁', inplace=True, errors='ignore')
            pretty_summ_df.drop(columns='F₂', inplace=True, errors='ignore')
            pretty_summ_df['exp'] = summ_df['exp']
            # pretty_summ_df = pretty_summ_df.set_index('exp')
            # print(pretty_summ_df.to_csv(float_format='%.1f'))
            print(pretty_summ_df.to_csv(index=False, float_format='%.1f'))
        else:
            cols = ['num_aligned_reads', 'target_acc', 'read_acc', 'err_far_ub',
                    'err_close_ub', 'err_only_ub', 'acc_xna', 'acc_pc', 'exp']
            print(summ_df[cols].to_csv(index=False, float_format='%.1f'))
    elif dict_print:
        pretty_summ_df.drop(columns='num_align', inplace=True)
        # pretty_summ_df.drop(columns='spec.', inplace=True, errors='ignore')
        pretty_summ_df['exp'] = summ_df['exp']
        print(pretty_summ_df.round(2).reset_index(drop=True).to_dict())
    else:
        cols = ['ub', '~ub_A', 'align'] # 'ub_A',
        for field in ['spec.','prec.','F₁','F₂']:
            if field in pretty_summ_df:
                cols.append(field)
        cols.append('exp')
        pretty_summ_df['exp'] = summ_df['exp']
        with pd.option_context("display.float_format", '{:.1f}'.format, 
                               # 'display.max_columns', None, # Not working as expected
                               'display.max_colwidth', None): # 92
            print(pretty_summ_df[cols])
    
    return pretty_summ_df
    

#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting comp_basecalls_perf - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    comp_basecalls_perf(**args)

    print('\n> Finished comp_basecalls_perf -', time.asctime( time.localtime(time.time()) ))
