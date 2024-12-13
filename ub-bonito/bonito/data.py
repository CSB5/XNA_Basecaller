import importlib
import os
from pathlib import Path

import numpy as np
# from torch.utils.data import DataLoader
from bonito.stitch_chunks import stitch_read, slice_xna, load_kmers_weight
from bonito.spike_chunks import spike_read, load_kmer_model

class ChunkDataSet:
    def __init__(self, chunks, targets, lengths, breakpoints=None, 
                 spike_kwargs=None, stitch_kwargs=None, epoch_reset_seed=False):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths
        self.breakpoints = breakpoints
        self.replace_6_letter = False
        
        if stitch_kwargs is not None:
            stitch_kwargs = stitch_kwargs.copy()
            # if not epoch_reset_seed: # Print only once, for training
            #     print(f"{stitch_kwargs=}")
            # print("Slicing XNA ctc-data (this might take some time)...")
            stitch_kwargs['xna_slices_df'] = slice_xna(
                stitch_kwargs['xna_ctc_dir'], stitch_kwargs['stitch_mode'], 
                verbose=False, include_chunks=True)
            
            ctc_dir = stitch_kwargs.pop('directory')
            if stitch_kwargs.pop('weighted_pos_pick'):
                stitch_kwargs['kmers_weight_df'] = load_kmers_weight(ctc_dir)
            
            stitch_kwargs.pop('xna_ctc_dir')
            # stitch_kwargs['equal_kmer_reps'] = False
        self.stitch_kwargs = stitch_kwargs

        if spike_kwargs is not None:
            spike_kwargs = spike_kwargs.copy()
            # if not epoch_reset_seed: # Print only once, for training
            #     print(f"{spike_kwargs=}")
            ref_filepath = spike_kwargs.pop('ref_filepath', None)
            spike_kwargs['kmer_poremodel'] = load_kmer_model(ref_filepath)
            # print("[WARNING] Forcing equal_kmer_reps=False")
            spike_kwargs['equal_kmer_reps'] = False
            # print("[WARNING] Forcing mix_ubs=False")
            # spike_kwargs['mix_ubs'] = False
        self.spike_kwargs = spike_kwargs
        
        if spike_kwargs is not None or stitch_kwargs is not None:
            self.epoch_reset_seed = epoch_reset_seed
            self.seed = 1910 if epoch_reset_seed else 2012 # change seeds train/val
            self.rng = np.random.default_rng(self.seed)

    def __getitem__(self, i):
        chunk, target, length = (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
        )
        
        #### Spiking on-the-fly (as data augmetantation)
        ### Skip if already contains UB (target > 4)
        # if self.spike_kwargs is not None and not any(target > 4):
        if self.spike_kwargs is not None or self.stitch_kwargs is not None:
            if i == 0 and self.epoch_reset_seed: 
                # Restore seed for validation consistency between batches
                # print("[WARNING] Reseting seed")
                self.rng = np.random.default_rng(self.seed)
            breakpts = self.breakpoints[i]
            
        if self.stitch_kwargs is not None:
            single_ctc_data = (chunk[0], target, length, breakpts)
            chunk, target, success = stitch_read(
                single_ctc_data, rng=self.rng, **self.stitch_kwargs)
            chunk = np.expand_dims(chunk, axis=0)
        
        if self.spike_kwargs is not None:
            chunk, target = spike_read(chunk[0], length, target, breakpts, 
                                       rng=self.rng, **self.spike_kwargs)
            chunk = np.expand_dims(chunk, axis=0)
        
        if self.replace_6_letter: # Workaround when using 5-letter models with Y base
            target[target==6] = 5
        
        return chunk, target, length

    def __len__(self):
        return len(self.lengths)


def load_script(directory, name="dataset", suffix=".py", **kwargs):
    directory = Path(directory)
    filepath = (directory / name).with_suffix(suffix)
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.Loader()
    return loader.train_loader_kwargs(**kwargs), loader.valid_loader_kwargs(**kwargs)


def load_numpy(limit, directory, spike_kwargs=None, stitch_kwargs=None):
    """
    Returns training and validation DataLoaders for data in directory.
    """
    train_data = load_numpy_datasets(limit=limit, directory=directory, 
        load_bkps=(spike_kwargs is not None or stitch_kwargs is not None))
    if os.path.exists(os.path.join(directory, 'validation')):
        valid_data = load_numpy_datasets(
            directory=os.path.join(directory, 'validation'),
            load_bkps=(spike_kwargs is not None or stitch_kwargs is not None)
        )
    else:
        print("[validation set not found: splitting training set (97%-3%)]")
        split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]
    
    train_loader_kwargs = {
        "dataset": ChunkDataSet(*train_data, spike_kwargs=spike_kwargs, 
                                stitch_kwargs=stitch_kwargs), 
        "shuffle": True}
    valid_loader_kwargs = {
        # "dataset": ChunkDataSet(*valid_data),
        "dataset": ChunkDataSet(*valid_data, spike_kwargs=spike_kwargs, 
                                stitch_kwargs=stitch_kwargs, epoch_reset_seed=True), 
        "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs


def load_numpy_datasets(limit=None, directory=None, load_bkps=False):
    """
    Returns numpy chunks, targets and lengths arrays.
    """
    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')

    indices = os.path.join(directory, "indices.npy")

    if os.path.exists(indices):
        print("[indices.npy found: using idx for subsampling]")
        idx = np.load(indices, mmap_mode='r')
        idx = idx[idx < lengths.shape[0]]
        if limit:
            idx = idx[:limit]
        
        if load_bkps:
            bkps = np.load(os.path.join(directory, "breakpoints.npy"), mmap_mode='r')
            return chunks[idx, :], targets[idx, :], lengths[idx], bkps[idx, :]
        
        return chunks[idx, :], targets[idx, :], lengths[idx]

    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]
    
    if load_bkps:
        bkps = np.load(os.path.join(directory, "breakpoints.npy"), mmap_mode='r')
        if limit:
            bkps = lengths[:bkps]
        return np.array(chunks), np.array(targets), np.array(lengths), np.array(bkps)
    
    return np.array(chunks), np.array(targets), np.array(lengths)
