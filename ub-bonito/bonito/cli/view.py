"""
Bonito model viewer - display a model architecture for a given config.
"""

import toml
import argparse
from bonito.util import load_symbol

import torch

def main(args):
    config = toml.load(args.config)
    # config["encoder"]['drop_rate'] = 0.50
    # config["encoder"]['drop_rate_bottom'] = 0.05
    
    Model = load_symbol(config, "Model")
    model = Model(config)
    print(model)
    print("Total parameters in model {:,d}".format(sum(p.numel() for p in model.parameters())))
    
    #### Extra Debug
    print("alphabet:", model.alphabet)
    print("n_base:", model.seqdist.n_base)
    print("state_len:", model.seqdist.state_len)

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("config")
    
    #### NN Architecture params
    parser.add_argument("-F", "--freeze-bottom", action="store_true", default=False)
    parser.add_argument("--num-unfreeze-top", default=1, type=int)
    parser.add_argument("--drop-rate", default=None, type=float)
    parser.add_argument("--drop-rate-bottom", default=None, type=float)
    parser.add_argument("--skip-top", action="store_true", default=False)

    return parser
