#!/usr/bin/env python
import pandas as pd
import argparse, sys, os, time
from glob import glob
from textwrap import fill
import itertools

TEST_ALIASES = {
    'xna20': 'XNA20-val_XNA20',
    'xna20_v2': 'XNA20-val_XNA20_v2',
    'xna16': 'A003-val_A003',
       '16': 'A003-val_A003',
    '4ds': 'A007-10k_filt-val',
    '1024': 'A026+A027-val',
    '1024-r200': 'A026+A027-val-r_200',
         'r200': 'A026+A027-val-r_200',
    '1024-4ds': 'A026+A027-val-4ds-r_700',
}

def load_args():
    ap = argparse.ArgumentParser(
        description='Consolidate UB validation. Read acc from different weights and selected best',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('model_dir',
        help='path to model directory, containing the basecalled validation.',
        nargs='+',
        type=str)
    
    ap.add_argument('-t', '--test_label',
        help=f"label of the test summary file to use. (e.g. {','.join(list(TEST_ALIASES.keys()))})",
        default='xna20_v2', type=str)
    ap.add_argument('-s', '--symlink_best',
        help='create symbolic link to best epoch, given target_metric.',
        action='store_true')
    ap.add_argument('-T', '--target_metric',
        help='target metric to be used when creating symbolic link.',
        choices=['err_only_ub','err_close_ub','err_far_ub','num_aligned_reads'],
        default='err_only_ub')
    
    ap.add_argument('-p', '--plot',
        help='plot main metrics from input train dirs.',
        action='store_true')
    
    ap.add_argument('-d', '--print-dict',
        help='plot main metrics from input train dirs.',
        action='store_true')
    
    ap.add_argument('-D', '--detailed_print',
        help='also print UB Acc per UB (X and Y).',
        action='store_true')
    
    # Optional arguments
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def consolidate_ub_validation(model_dir, test_label, symlink_best=False,
                              target_metric='err_only_ub', plot=False, 
                              print_dict=False, detailed_print=False):
    if isinstance(model_dir, list) and len(model_dir) == 1:
        model_dir = model_dir[0]
    else:
        val_summs_df = []
        for single_model_dir in model_dir:
            try:
                val_summ_df = consolidate_ub_validation(
                    [single_model_dir], test_label, target_metric=target_metric,
                    detailed_print=detailed_print, symlink_best=symlink_best)
                if val_summ_df is not None:
                    val_summs_df.append(val_summ_df)
            except FileNotFoundError:
                print("  [WARNING]: Skipping dir because results summary file was not found.")
        
        if val_summs_df == []:
            print("Exiting: No valid file found!")
            return
        
        val_summs_df = pd.concat(val_summs_df)
        if print_dict:
            # PRINT_COLS = ['exp','epoch','err_far_ub']
            PRINT_COLS = ['exp','epoch','align','err_only_ub','err_far_ub']
            print(val_summs_df[PRINT_COLS].reset_index(drop=True).to_dict())
        
        #### Plotting
        if plot:
            print()
            val_summs_df['exp'] = val_summs_df.exp.str.wrap(80)
            val_summs_df['ub_acc'] = 100-val_summs_df.err_only_ub
            val_summs_df['nat_base_acc'] = 100-val_summs_df.err_far_ub
            
            plot_col = 'ub_acc'
            if val_summs_df.ub_acc.max() == 0:
                plot_col = 'nat_base_acc'
            
            print("Loading pyplot... (might take a while)")
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("Plotting...")
            with sns.axes_style("whitegrid"): # whitegrid | ticks | darkgrid
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=False)

            available_markers = itertools.cycle(('o', '^', 's', 'P', '*','X'))
            markers = [ next(available_markers) for _ in range(val_summs_df.exp.nunique()) ]
            if not detailed_print:
                sns.pointplot(data=val_summs_df, x='epoch', y=plot_col, hue='exp', 
                              palette='colorblind', markers=markers, ax=axes[0])
            else:
                melt_val_summs_df = pd.melt(val_summs_df, 
                        id_vars=['epoch','exp'], 
                        value_vars=['ub_acc','ub_X','ub_Y'], 
                        # value_vars=['ub_X','ub_Y'], 
                        var_name='ubs', 
                        value_name='ubs_acc')
                sns.lineplot(data=melt_val_summs_df, x='epoch', y='ubs_acc', hue='exp',
                    style='ubs', palette='colorblind', 
                    markers=('o', 'X', '$\mathbf{Y}$'), # markers=True, 
                    lw=2.5, markersize=10, ax=axes[0])
                axes[0].set(xticks=melt_val_summs_df.epoch.unique())
                axes[0].grid(visible=False, which='major', axis='x')

            sns.pointplot(data=val_summs_df, x='epoch', y='align', hue='exp', 
                          palette='colorblind', markers=markers, ax=axes[1])
            
            for ax in axes:
                ax.legend().set_draggable(True)
                plt.setp(ax.get_legend().get_texts(), fontsize=8) # for legend text
                # sns.move_legend(ax, 'best',  wrap=True)
            
            fig.suptitle(f'{test_label=}')
            
            plt.tight_layout()
            print("Showing...")
            plt.show()
        
        return val_summs_df
    print(f">>> model_dir: {model_dir}")
    
    test_label = TEST_ALIASES.get(test_label, test_label)
    print(f"> Results file: results_summ-{test_label}.csv")
    
    basecall_dirs = glob(os.path.join(model_dir, 'basecalls-weights_*'))
    if len(basecall_dirs) == 0:
        raise FileNotFoundError(os.path.join(model_dir, 'basecalls-weights_*'))
    
    #### Read summary csv files
    val_summ_dfs = []
    for basecall_dir in basecall_dirs:
        val_summ_filepath = os.path.join(basecall_dir, f"results_summ-{test_label}.csv")
        if not os.path.exists(val_summ_filepath):
            # print(f'  [WARNING]: Results file not found, skipping dir: {basecall_dir}')
            continue
        #### TODO add a max bad row if no result found?
        val_summ_df = pd.read_csv(val_summ_filepath)
        val_summ_df['weight'] = os.path.basename(basecall_dir).split('-')[-1]
        val_summ_df['epoch'] = int(os.path.basename(basecall_dir).split('-')[-1].split('_')[-1])
        val_summ_df = val_summ_df.set_index('epoch', drop=False)
        
        
        #### Detailed reading UB Accs per UB
        if detailed_print:
            detailed_summ_filepath = os.path.join(basecall_dir,
                f"results_summ-{test_label}-by_tar.csv")            
            results_by_tar_df = pd.read_csv(detailed_summ_filepath)
            results_by_tar_df = results_by_tar_df.set_index(['target_id','strand'])
            
            for ub, strand in [('X','F'),('Y','R')]:
                if strand in results_by_tar_df.index.get_level_values('strand'):
                    val_summ_df[f'ub_{ub}'] = results_by_tar_df.xs(
                        strand, level="strand").ub_acc.mean()
        
        val_summ_dfs.append(val_summ_df)
    if len(val_summ_dfs) == 0:
        print('[WARNING]: No files found, exiting...')
        return
    val_summ_df = pd.concat(val_summ_dfs).sort_index().round(1)
    
    # print(val_summ_df[['err_only_ub','err_close_ub','err_far_ub','num_aligned_reads']])
    pretty_summ_df = 100-val_summ_df[['err_only_ub','err_close_ub','err_far_ub']]
    pretty_summ_df.columns = ['ub', 'ub_A', '~ub_A']
    for field in ['demux','align','precision','f1_score']:
        if field in val_summ_df:
            pretty_summ_df[field] = val_summ_df[field]
    pretty_summ_df.rename(columns={'specificity':'spec.','precision':'prec.',
                                   'f1_score':'F1-scr'}, 
                          inplace=True)
    pretty_summ_df['N_align'] = val_summ_df['num_aligned_reads']
    
    if detailed_print:
        for ub in ['X','Y']:
            pretty_summ_df[f'ub_{ub}'] = val_summ_df[f'ub_{ub}']
    
    #### Print results per epoch
    with pd.option_context("display.float_format", '{:.1f}'.format):
        print(pretty_summ_df)
    
    best_df = pd.concat(
        (val_summ_df[['err_only_ub', 'err_close_ub','err_far_ub']].idxmin(),
         val_summ_df[['num_aligned_reads']].idxmax())
        )
    print("Best epoch for each metric:")
    print(best_df.to_frame().T)
    
    if target_metric.startswith('err'):
        best_epoch = val_summ_df[target_metric].idxmin()
    else:
        best_epoch = val_summ_df[target_metric].idxmax()
    best_metric = val_summ_df.loc[best_epoch, target_metric]
    best_epochs_df = val_summ_df[val_summ_df[target_metric]==best_metric]
    len(best_epochs_df)
        
    if len(best_epochs_df) == 1:
        # print(f"Best epoch for {target_metric=} is {best_epoch=} with {best_metric}")
        print(f"Best epoch is {best_epoch} with {target_metric}={best_metric}")
    else:
        print("Multiple epochs according to selected target_metric, refining for best 'err_far_ub'.")
        best_epoch = best_epochs_df.err_far_ub.idxmin()
        best_err_far_ub = val_summ_df.loc[best_epoch, 'err_far_ub']
        print(f"Best epoch is {best_epoch} with {target_metric}={best_metric} and err_far_ub={best_err_far_ub}")
    
    #### Find best epoch and create symlinks
    if symlink_best:
        
        best_weights = f'weights_{best_epoch}.tar'
        # symlink_epoch = val_summ_df.epoch.max() + 1
        symlink_epoch = 99 # Set high value and improbable, to avoid confusion
        symlink_weights = f'weights_{symlink_epoch}.tar'
        symlink_weights_filepath = os.path.join(model_dir, symlink_weights)
        # print(f"Looking for best epoch for {target_metric=}")
        if os.path.exists(symlink_weights_filepath) and os.path.islink(symlink_weights_filepath):
            old_symlink_weights_filepath = os.readlink(symlink_weights_filepath)
            # print(f"> > Unlinking: {old_symlink_weights_filepath=}")
            print(f"> > Unlinking: '{symlink_weights_filepath}' -> '{old_symlink_weights_filepath}'")
            old_best_weights = old_symlink_weights_filepath[:-4]
            os.unlink(symlink_weights_filepath)
        else:
            old_best_weights = f'weights_{val_summ_df.epoch.max()}'
            
        ### Always create symlink? R: Yes, since now I do it for epoch 99
        # if best_epoch != val_summ_df.epoch.max():
        if True:
            print(f"> Creating symlink: '{symlink_weights_filepath}' -> '{best_weights}'")
            os.symlink(best_weights, symlink_weights_filepath)
        else:
            print("> Skipping symlink because last epoch is the best.")
        
        basecalls_dir = os.path.join(model_dir, 'basecalls')
        best_basecalls_dir = 'basecalls-' + best_weights[:-4]
        os.makedirs(os.path.join(model_dir, best_basecalls_dir), exist_ok=True)
        # if not os.path.exists( os.path.join(model_dir, best_basecalls_dir)):
            # os.makedirs( os.path.join(model_dir, best_basecalls_dir))
        if os.path.exists(basecalls_dir):
            if os.path.islink(basecalls_dir):
                if os.readlink(basecalls_dir) != best_basecalls_dir:
                    print(f"> > Unlinking: '{basecalls_dir}' -> '{os.readlink(basecalls_dir)}'")
                    os.unlink(basecalls_dir)
                    print(f"> Creating symlink: '{basecalls_dir}' -> '{best_basecalls_dir}'")
                    os.symlink(best_basecalls_dir, basecalls_dir)
                else:
                    print("> Skipping basecalls symlink because already points to best epoch.")
            else:
                ### Move/rename dir if not symlink
                old_best_basecalls_dir = os.path.join(model_dir, 'basecalls-'+old_best_weights)
                if os.path.exists(old_best_basecalls_dir):
                    old_best_basecalls_dir = os.path.join(model_dir, 
                                                          'REPEAT-basecalls-'+old_best_weights)
                print(f"> > Renaming: '{basecalls_dir}' -> '{old_best_basecalls_dir}'")
                os.rename(basecalls_dir, old_best_basecalls_dir)
                print(f"> Creating symlink: '{basecalls_dir}' -> '{best_basecalls_dir}'")
                os.symlink(best_basecalls_dir, basecalls_dir)
        else:
            print(f"> Creating symlink: '{basecalls_dir}' -> '{best_basecalls_dir}'")
            os.symlink(best_basecalls_dir, basecalls_dir)
    
    # train_summ_dir = os.path.dirname(model_dir)
    train_summ_dir, train_summ_base = os.path.split(model_dir)
    if train_summ_dir == '':
        train_summ_dir = model_dir
    if train_summ_base != 'training' and train_summ_base != '':
        train_summ_dir = train_summ_base
    val_summ_df['exp'] = train_summ_dir
    # val_summ_df['exp'] = fill(train_summ_dir, 80) # Wrap text
    
    # return val_summ_df.loc[best_epoch]
    return val_summ_df
    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    # TODO Replace XXXs and main() name
    print('> Starting Consolidate UB validation - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    consolidate_ub_validation(**args)

    print('\n> Finished Consolidate UB validation -', time.asctime( time.localtime(time.time()) ))
