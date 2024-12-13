"""
Bonito CRF basecalling
"""

import sys

import torch
import numpy as np
from koi.decode import beam_search, to_str

from bonito.multiprocessing import thread_iter
from bonito.util import chunk, stitch, batchify, unbatchify, half_supported


def stitch_results(results, length, size, overlap, stride, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: stitch_results(v, length, size, overlap, stride, reverse=reverse)
            for k, v in results.items()
        }
    return stitch(results, size, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
    Compute scores for model.
    """
    use_viterbi = model.encoder[-1].expand_blanks # Proxy for use_koi == False
    # sys.stderr.write(f"> use_viterbi : {use_viterbi}\n")
    if not use_viterbi: 
        #### [my-mod] Using beam search
        # sys.stderr.write("> Using beam search decoding.\n")
        with torch.inference_mode():
            device = next(model.parameters()).device
            dtype = torch.float16 if half_supported() else torch.float32
            scores = model(batch.to(dtype).to(device))
            if reverse:
                scores = model.seqdist.reverse_complement(scores)
    
            sequence, qstring, moves = beam_search(
                scores, beam_width=beam_width, beam_cut=beam_cut,
                scale=scale, offset=offset, blank_score=blank_score
            )
    else:
        #### [my-mod] Using viterbi decoding
        # sys.stderr.write("> Using viterbi decoding.\n")
        with torch.no_grad():
            device = next(model.parameters()).device
            dtype = torch.float16 if half_supported() else torch.float32
            scores = model(batch.to(dtype).to(device))
            if reverse:
                scores = model.seqdist.reverse_complement(scores)

            sequence = model.decode_batch(scores)
            ### String seq: GAGGACCAGTGATACCATG 
            ### Converting sequence to: [ 78, 65, ...]
            sequence = [ [ ord(c) for c in seq ] for seq in sequence ]
            
            chunk_size, batch_size, _ = scores.shape
            pad_sequence = np.zeros((batch_size, chunk_size), dtype=np.int)
            qstring = np.zeros((batch_size, chunk_size))
            for idx, seq in enumerate(sequence):
                pad_sequence[idx, :len(seq)] = seq
                qstring[idx, :len(seq)] = ord('O') # Dummy quality: middle value
            
            sequence = np.asarray(pad_sequence)
            
            moves = np.zeros((batch_size, chunk_size)) # Dummy moves
            
            ### torch.tensor needed by stitch
            sequence = torch.tensor(sequence, dtype=torch.int8)
            qstring = torch.tensor(qstring, dtype=torch.int8)
            moves = torch.tensor(moves)
            
    return {
        'qstring': qstring,
        'sequence': sequence,
        'moves': np.array(moves, dtype=bool),
    }


def apply_stride_to_moves(model, attrs):
    moves = np.array(attrs['moves'], dtype=bool)
    sig_move = np.full(moves.size * model.stride, False)
    sig_move[np.where(moves)[0] * model.stride] = True
    return {
        'qstring': to_str(attrs['qstring']),
        'sequence': to_str(attrs['sequence']),
        'sig_move': sig_move,
    }


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32, reverse=False):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    scores = thread_iter(
        (read, compute_scores(model, batch, reverse=reverse)) for read, batch in batches
    )

    results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return thread_iter(
        (read, apply_stride_to_moves(model, attrs))
        for read, attrs in results
    )
