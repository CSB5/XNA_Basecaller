#!/usr/bin/env python3

"""
Bonito training.
"""

import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module

from bonito.data import load_numpy, load_script
from bonito.util import __models__, default_config, default_data
from bonito.util import load_model, load_symbol, init, half_supported
from bonito.training import load_state, Trainer

import toml
import torch
import numpy as np
from torch.utils.data import DataLoader


def main(args):
    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    init(args.seed, args.device)
    device = torch.device(args.device)
    
    #### Loading data
    print("[loading data]")
    num_workers = 4
    stitch_kwargs = spike_kwargs = None
        
    if args.stitch_mode and args.prop_ubs > 0:
        #### Setting Stitch params
        num_workers = 12
        DEF_XNA_CTC_DIR = os.path.expanduser("./ub-bonito/bonito/data/xna_r9.4.1")
        xna_ctc_dir = args.xna_ctc_dir if args.xna_ctc_dir is not None else DEF_XNA_CTC_DIR
        stitch_kwargs = dict( # verbose=False, kmer_len=6, 
            ubs=args.ubs, 
            prop_ubs=(args.prop_ubs-args.synth_prop_ubs), #prop_ubs=args.prop_ubs, 
            var_prop_ubs=args.var_prop_ubs,
            stitch_mode=args.stitch_mode,
            cand_sample_size=args.cand_sample_size, xna_ctc_dir=xna_ctc_dir,
            weighted_pos_pick=args.weighted_pos_pick, directory=args.directory,
            noise_std=args.stitch_noise_std, noise_mode=args.stitch_noise_mode,
            permute_win_size=args.permute_win_size, 
            pad=args.ub_pad,
        )
        print(f"{stitch_kwargs=}")
            
        
    if args.spike and args.prop_ubs > 0:
        #### Setting Synthetic spike params
        num_workers = 8
        KMER_MODELS_DIR = os.path.expanduser('./ub-bonito/bonito/data/')
        DEF_KMER_MODEL = 'r9.4_450bps.nucleotide.6mer.XNA-Px_Ds.template.model'
        ref_filepath = (args.ref_filepath if args.ref_filepath is not None
                        else os.path.join(KMER_MODELS_DIR, DEF_KMER_MODEL))
        spike_kwargs = dict(
            ubs=args.ubs, prop_ubs=args.prop_ubs, var_prop_ubs=args.var_prop_ubs,
            noise_std=args.noise_std, variable_noise=args.variable_noise,
            fully_synth=args.fully_synth,
            ref_filepath=ref_filepath, std_dist=args.std_dist,
            pad=args.ub_pad,
        )
        print(f"{spike_kwargs=}")
        
    try:
        train_loader_kwargs, valid_loader_kwargs = load_numpy(args.chunks, 
            args.directory, spike_kwargs=spike_kwargs, stitch_kwargs=stitch_kwargs)
        ### Testing data loading
        # train_loader_kwargs['dataset'][0]
        # valid_loader_kwargs['dataset'][0]
    except FileNotFoundError as e:
        print("> Error while loading numpy ctc-data: ")
        raise e
        print(e)
        print(">> Loading data via 'load_script()': ")
        train_loader_kwargs, valid_loader_kwargs = load_script(
            args.directory, seed=args.seed, chunks=args.chunks,
            valid_chunks=args.valid_chunks)
    
    if args.num_workers is not None:
        num_workers = args.num_workers
    print(f"> num_workers: {num_workers}")
    loader_kwargs = {
        "batch_size": args.batch, 
        "num_workers": num_workers, # added more workers due to data-aug and truncnorm [n_proc,processes]
        "pin_memory": True
    }
    train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs)
    valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs)

    #### Loading model
    if args.pretrained:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        config_file = os.path.join(dirname, 'config.toml')
    else:
        config_file = args.config

    config = toml.load(config_file)

    argsdict = dict(training=vars(args))

    os.makedirs(workdir, exist_ok=True)
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    print("[loading model]")
    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        # model = load_model(args.pretrained, device, half=False)
        
        model = load_model(args.pretrained, device, half=False, 
                           skip_top=args.skip_top, drop_rate=args.drop_rate, 
                           drop_rate_bottom=args.drop_rate_bottom)
    else:
        model = load_symbol(config, 'Model')(config)
    
    if model.seqdist.n_base == 5 and model.seqdist.alphabet[-1]=='Y': 
        # XNA, but only 5-letter. Need to replace 6th letter (Y) at ctc-data.
        train_loader.dataset.replace_6_letter = valid_loader.dataset.replace_6_letter = True
    
    #### Freeze layers weights
    print(f"> drop_rate: {args.drop_rate} | drop_rate_bottom: {args.drop_rate_bottom}")
    # Freeze weights from bottom layers, unfreeze from top/last
    if args.freeze_bottom:
        print("[WARNING: freezing bottom layers]")
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"> > Unfreezing top/last {args.num_unfreeze_top} layer(s):")
        unfreeze_count = 0
        unfreeze_layer_names = []
        unfreeze_layer_idxs = []
        layer_idxs = list(range(len(model.encoder)))
        for layer_id in layer_idxs[::-1]:
            layer = model.encoder[layer_id]
            if unfreeze_count < args.num_unfreeze_top:
                named_parameters = list(layer.named_parameters())
                if len(named_parameters) > 0:
                    for param_name, param in named_parameters:
                        param.requires_grad = True
                    unfreeze_layer_names.append(layer.name)
                    unfreeze_layer_idxs.append(layer_id)
                    unfreeze_count += 1
            else:
                if isinstance(layer, torch.nn.Dropout) and (layer_id+1) not in unfreeze_layer_idxs:
                    # layer.eval() # Did not worked, Trainer calls "self.model.train()"
                    layer.p = 0 # Deactivating dropout by setting rate to 0
        print("> >", ', '.join(unfreeze_layer_names))
    
    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None
    
    #### Build trainer
    trainer = Trainer(
        model, device, train_loader, valid_loader,
        use_amp=half_supported() and not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split
    )
    
    print(f"> batch_size: {args.batch} | epochs: {args.epochs} | lr: {args.lr}")
    
    trainer.fit(workdir, args.epochs, args.lr)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=default_config)
    group.add_argument('--pretrained', default="")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=0, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    
    parser.add_argument("--num-workers", type=int)
    
    #### NN Architecture params
    parser.add_argument("-F", "--freeze-bottom", action="store_true", default=False)
    parser.add_argument("--num-unfreeze-top", default=1, type=int)
    parser.add_argument("--drop-rate", default=None, type=float)
    parser.add_argument("--drop-rate-bottom", default=None, type=float)
    parser.add_argument("--skip-top", action="store_true", default=False)
    
    #### [spike-aug] Synthetic Signal params
    parser.add_argument('--spike',
        help="Spike chunks with synthetic XNA",
        action='store_true')
    parser.add_argument('--prop-ubs',
        help="proportion of bases to be spiked with UBs (Ex: 0.01 = 1%%).",
        type=float, default=0)
    parser.add_argument('--var-prop-ubs',
        help="variable UB proportion with prop_ub+-var_prop_ub (Ex: 5%%-4%% to 5%%+4%%).",
        type=float)
    parser.add_argument('--ubs', 
        default=['X','Y'], type=list,
        help='List of UBs to spike in (X, Y, XY). Set N to spike same DNA sequence.')
    parser.add_argument('--noise-std',
        help="Set stdev of noise to be added (0 means none).",
        type=float, default=0)
    parser.add_argument('--ref-filepath',
        help="filepath to reference kmer model template for synthetic signal generation.", 
        type=str)
    parser.add_argument('--variable-noise',
        help="Chooses a different noise std for each UB position (max: --noise_std).",
        action='store_true')
    parser.add_argument('--std-dist',
        help="How to sample stdev for synthetic signals.",
        type=str, default='uniform')
    parser.add_argument('--fully_synth',
        help="Synthesize full target, XNA and DNA kmers.",
        action='store_true')
    parser.add_argument('--synth-prop-ubs',
        help="If using both stitch and spike, set UB prop. for synth/spike.",
        type=float, default=0)
    
    #### Stitch params
    parser.add_argument('--stitch-mode',
        help="Mode for stitching xna slices: entire 6-mer slice or per kmer.",
        choices=['per_kmer','per_slice','mixed'],
        type=str)
    parser.add_argument('--cand-sample-size',
        help="Number of XNA slice candidates to sample, before picking slice w/ closest chunk len.",
        type=int, default=10)
    parser.add_argument('--weighted-pos-pick', dest='weighted_pos_pick',
        help="Number of XNA slice candidates to sample, before picking slice w/ closest chunk len.",
        action='store_true')
    parser.add_argument('--xna_ctc_dir',
        help="Path to ctc-data dir of source XNA to be sliced and stitched.",
        type=str)
    parser.add_argument('--permute-win-size',
        help="Stitch data augmentation through permutation of signals within set win size",
        type=int, default=0)
    parser.add_argument('--stitch-noise-std',
        help="Set stdev of noise to be added to stitched (0 means none).",
        type=float, default=0)
    parser.add_argument('--stitch-noise-mode',
        help="Stitched noise mode to be used.",
        choices=['single','single_variable','block_add','block_mult'],
        type=str, default='single')
    
    parser.add_argument('--ub-pad',
        help="Number of bases to pad between selection of UBs.",
        type=int, default=5)
    
    return parser
