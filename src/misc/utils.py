import os
import re
import pandas as pd
import numpy as np
import multiprocessing.dummy as mp

import Levenshtein
from Bio import Align

from misc.data_io import get_read_seq, index_reads_file, get_read_qual

# Source: from taiyaki.bio import complement, reverse_complement
# _COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N',
#                'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
#                '-': '-'}

### Using Z instead of X
# _COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N',
#                'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
#                '-': '-',
#                'Z': 'Y', 'Y': 'Z',
#                'z': 'y', 'y': 'z',
#                }

### Using X (needed to remove orig x complement)
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n',
               '-': '-',
               'X': 'Y', 'Y': 'X',
               'x': 'y', 'y': 'x',
               }

def complement(seq, compdict=_COMPLEMENT):
    """ Return complement of a base sequence.

    Args:
        seq (str): Base sequence
        compdict (dict, optional): Base complements

    Returns:
        String of complemented bases.
        
    Source: from taiyaki.bio import complement, reverse_complement
    """
    return ''.join(compdict[b] for b in seq)

def reverse_complement(seq, compdict=_COMPLEMENT):
    """ Return reverse complement of a base sequence.

    Args:
        seq (str): Base sequence
        compdict (dict, optional): Base complements

    Returns:
        String of reverse complemented bases.
        
    Source: from taiyaki.bio import complement, reverse_complement
    """
    return complement(seq, compdict)[::-1]

# def get_ref(read_info, ref_db, replace_N='Z'):
def get_ref(ref_id, ref_db, strand, replace_N='X'):
    
    # ref_id = (read_info.target_id if 'target_id' in read_info else 
    #           read_info.barcode_name)
    
    read_ref = ref_db[ref_id].upper()
    
    # Replacing N by selected value if necessary
    if replace_N is not None:
        read_ref = read_ref.replace('N', replace_N)

    # if read_info.strand == '-':
    if strand == '-':
        #  Mapped to reverse strand
        read_ref = reverse_complement(read_ref)
    
    # TODO is the start-end trimming necessary? Before or after? 
    # It seems after for my case, though it was before for get_refs_from_sam.py
    # pad = 0
    # start = max(0, read_info.read_start - pad)
    # end = min(len(read_ref), read_info.read_end + pad)
    # read_ref = read_ref[start:end]
    
    return read_ref

def parse_cs_flag(cs, 
         cs_regex=':[0-9]+|\*[a-z]{2}|[=+-][A-Za-z]+|~[a-z]{2}[0-9]+[a-z]{2}'):
    """
    Parsing cs flag from .paf generated with minimap.

    Parameters
    ----------
    cs : str
        cs string.
    cs_regex : string, optional
        Regular expression for parsing cs flag string.
        The default is ':[0-9]+|\*[a-z]{2}|[=+-][A-Za-z]+|~[a-z]{2}[0-9]+[a-z]{2}'.

    Returns
    -------
    operations : list of strings
        List of operations.

    """
    p = re.compile(cs_regex)
    
    operations = p.findall(cs)
    
    return operations

def compute_read_matches(read_seq, operations=None, align_start=None,
                         align_end=None, target_length=None, read_info=None):
    """
    Compute which positions in read/query are matched with target during minimap alignment.
    Return read alignment with respect to target, but with query substitutions.
    Based on paf extract operations from cs flag.
    https://lh3.github.io/minimap2/minimap2.html#10

    Parameters
    ----------
    read_seq : str
        read/query aligned sequence.
    operations : list of str
        List of operations provided by cs flag.
        Output of parse_cs_flag().
    align_start : int
        Target alignment start position.
    align_end : int
        Target alignment end position.
    target_length : int
        Target length.

    Raises
    ------
    NotImplementedError
        op_symbol == ~.

    Returns
    -------
    read_matches : ndarray of chars.
        DESCRIPTION.

    """
    if read_info is not None:
        operations = parse_cs_flag(read_info.cs)
        align_start, align_end, target_length = read_info[['target_start',
                                       'target_end','target_length']].values
    
    read_seq_arr = list(read_seq)

    ### Idea is to append the read_matches list depending on the operations

    # Start and finish by appeding the missed aligned values -
    read_matches = ['-'] * align_start

    pointer = 0

    for op in operations:
        op_symbol = op[0]
        op_val = op[1:]
        
        if op_symbol == '=': # Identical sequence (long form)
            # print(read_seq[pointer:pointer+len(op_val)])
            read_matches += read_seq_arr[pointer:pointer+len(op_val)]
            pointer += len(op_val)
        elif op_symbol == ':': # Identical sequence length (short form)
            read_matches += read_seq_arr[pointer:pointer+int(op_val)]
            pointer += int(op_val)
        elif op_symbol == '*':
            # print(read_seq_arr[pointer].upper() , op_val[1].upper())
            assert read_seq_arr[pointer].upper().replace('X','N').replace('Y','N') == op_val[1].upper()
            # assert read_seq_arr[pointer].upper() == op_val[1].upper()
            read_matches += [ read_seq_arr[pointer] ]
            pointer += 1
        elif op_symbol == '+':
            # print((''.join(read_seq_arr[pointer:pointer+len(op_val)]).upper(), op_val.upper()))
            ### Bases in read/query that can be skipped
            assert (''.join(read_seq_arr[pointer:pointer+len(op_val)]).upper().replace('X','N').replace('Y','N')
                    == op_val.upper())
            pointer += len(op_val)
        elif op_symbol == '-':
            ### Bases in target that are missed, no movement in read/query
            read_matches += ['-'] * len(op_val)
        elif op_symbol == '~':
            raise NotImplementedError()
            pass
    
    read_matches += ['-'] * (target_length - align_end)
    
    return read_matches

def compute_read_matches_query(read_seq, operations=None, read_info=None):
    """
    Compute which positions in read/query are matched with target during minimap alignment.
    Return read alignment with respect to query, but with target substitutions.
    Based on paf extract operations from cs flag.
    https://lh3.github.io/minimap2/minimap2.html#10

    Parameters
    ----------
    read_seq : str
        read/query aligned sequence.
    operations : list of str
        List of operations provided by cs flag.
        Output of parse_cs_flag().

    Raises
    ------
    NotImplementedError
        op_symbol == ~.

    Returns
    -------
    read_matches : ndarray of chars.
        DESCRIPTION.

    """
    if read_info is not None:
        operations = parse_cs_flag(read_info.cs)
        # print(f"{operations=}")
    
    read_seq_arr = list(read_seq)

    read_matches = []

    pointer = 0
    for op in operations:
        op_symbol = op[0]
        op_val = op[1:]
        
        if op_symbol == '=': # Identical sequence (long form)
            # print(read_seq[pointer:pointer+len(op_val)])
            read_matches += read_seq_arr[pointer:pointer+len(op_val)]
            pointer += len(op_val)
        elif op_symbol == ':': # Identical sequence length (short form)
            read_matches += read_seq_arr[pointer:pointer+int(op_val)]
            pointer += int(op_val)
        elif op_symbol == '*': # Substitution: ref to query
            # print(read_seq_arr[pointer].upper() , op_val[1].upper())
            assert read_seq_arr[pointer].upper().replace('X','N').replace('Y','N') == op_val[1].upper()
            # assert read_seq_arr[pointer].upper() == op_val[1].upper()
            read_matches.append(op_val[0].upper())
            pointer += 1
        elif op_symbol == '+': # Insertion to the reference
            ### Bases only in read/query
            # print((''.join(read_seq_arr[pointer:pointer+len(op_val)]).upper(), op_val.upper()))
            assert (''.join(read_seq_arr[pointer:pointer+len(op_val)]).upper().replace('X','N').replace('Y','N')
                    == op_val.upper())
            read_matches += ['-'] * len(op_val)
            pointer += len(op_val)
        elif op_symbol == '-': # Deletion from the reference
            ### Bases in target that are missed, no movement in read/query
            # read_matches += list(op_val)
            pass
        elif op_symbol == '~':
            raise NotImplementedError()
            pass
    
    return read_matches

def compute_alignments(read_seq, target, operations=None, align_start=None,
                         align_end=None, target_length=None, read_info=None):
    """
    Compute which positions in read/query are matched with target during minimap alignment.
    Return read alignment with respect to target, but with query substitutions.
    Based on paf extract operations from cs flag.
    https://lh3.github.io/minimap2/minimap2.html#10

    Parameters
    ----------
    read_seq : str
        read/query aligned sequence.
    operations : list of str
        List of operations provided by cs flag.
        Output of parse_cs_flag().
    align_start : int
        Target alignment start position.
    align_end : int
        Target alignment end position.
    target_length : int
        Target length.

    Raises
    ------
    NotImplementedError
        op_symbol == ~.

    Returns
    -------
    read_matches : ndarray of chars.
        DESCRIPTION.

    """
    if read_info is not None:
        operations = parse_cs_flag(read_info.cs)
        align_start, align_end, target_length = read_info[['target_start',
                                       'target_end','target_length']].values
    # print(f"{operations=}")
    
    read_seq_arr = list(read_seq)
    target_arr = list(target)

    ### Idea is to append the read_matches list depending on the operations

    # Start and finish by appending the missed aligned values -
    read_aligned = ['-'] * align_start
    target_aligned = target_arr[:align_start]

    pointer = 0 # read/query
    target_pointer = align_start
    # print(f"{len(read_aligned)=}")
    # print(f"{len(target_aligned)=}")
    # print(f"{pointer=}")
    # print(f"{target_pointer=}")

    for op in operations:
        # print(f"{op=}")
        op_symbol = op[0]
        op_val = op[1:]
        
        if op_symbol == '=': # Identical sequence (long form)
            # print(read_seq[pointer:pointer+len(op_val)])
            read_aligned += read_seq_arr[pointer:pointer+len(op_val)]
            target_aligned += read_seq_arr[pointer:pointer+len(op_val)]
            pointer += len(op_val)
            target_pointer += len(op_val)
        elif op_symbol == ':': # Identical sequence length (short form)
            read_aligned += read_seq_arr[pointer:pointer+int(op_val)]
            target_aligned += read_seq_arr[pointer:pointer+int(op_val)]
            pointer += int(op_val)
            target_pointer += int(op_val)
        elif op_symbol == '*': # Substitution: ref to query
            # print(read_seq_arr[pointer].upper() , op_val[1].upper())
            assert read_seq_arr[pointer].upper().replace('X','N').replace('Y','N') == op_val[1].upper()
            # assert read_seq_arr[pointer].upper() == op_val[1].upper()
            read_aligned += [ read_seq_arr[pointer] ]
            target_aligned.append(op_val[0].upper())
            pointer += 1
            target_pointer += 1
        elif op_symbol == '+': # Insertion to the reference
            # print((''.join(read_seq_arr[pointer:pointer+len(op_val)]).upper(), op_val.upper()))
            ### Bases in read/query that can be skipped
            assert (''.join(read_seq_arr[pointer:pointer+len(op_val)]).upper().replace('X','N').replace('Y','N')
                    == op_val.upper())
            read_aligned += read_seq_arr[pointer:pointer+len(op_val)]
            pointer += len(op_val)
            target_aligned += ['-'] * len(op_val)
        elif op_symbol == '-': # Deletion from the reference
            ### Bases in target that are missed, no movement in read/query
            read_aligned += ['-'] * len(op_val)
            # target_aligned += target[target_pointer:target_pointer+len(op_val)]
            target_aligned += target_arr[target_pointer:target_pointer+len(op_val)]
            target_pointer += len(op_val)
        elif op_symbol == '~': # Intron length and splice signal
            raise NotImplementedError()
            pass
    
    #     print(f"{len(read_aligned)=}")
    #     print(f"{len(target_aligned)=}")
    #     print(f"{pointer=}")
    #     print(f"{target_pointer=}")
    #     print("++++++++++++++++++")
    # print(f"{target_length=}")
    # print(f"{align_end=}")
    
    read_aligned += ['-'] * (target_length - align_end)
    target_aligned += target_arr[target_pointer:]
    
    # print(f"{len(read_aligned)=}")
    # print(f"{len(target_aligned)=}")
        
    assert len(read_aligned) == len(target_aligned)
    
    return read_aligned, target_aligned

def compute_target_matches(target, operations, align_start, align_end):
    """
    Compute which positions in target are matched during minimap alignment.
    Based on paf extract operations from cs flag.
    https://lh3.github.io/minimap2/minimap2.html#10

    Parameters
    ----------
    target : str
        target/reference sequence.
    operations : list of str
        List of operations provided by cs flag.
        Output of parse_cs_flag().
    align_start : int
        Alignment start position.
    align_end : int
        Alignment end position.

    Raises
    ------
    NotImplementedError
        op_symbol == ~.

    Returns
    -------
    target_matches : ndarray of chars.
        DESCRIPTION.

    """
    
    target_matches = np.asarray(list(target))
    target_matches[:align_start] = '-'
    target_matches[align_end:] = '-'
    pointer = align_start
    
    for op in operations:
        op_symbol = op[0]
        op_val = op[1:]
        
        if op_symbol == '=': # Identical sequence (long form)
            pointer += len(op_val)
        elif op_symbol == ':': # Identical sequence length
            pointer += int(op_val)
        elif op_symbol == '*':
            # print(target_matches[pointer], op_val[0])
            # print(target_matches[pointer-1:pointer+2], op_val)
            assert target_matches[pointer].upper() == op_val[0].upper()
            target_matches[pointer] = '*'
            pointer += 1
        elif op_symbol == '+':
            pass
        elif op_symbol == '-':
            # print(''.join(target_matches[pointer:pointer+len(op_val)]).upper(), op_val.upper())
            assert (''.join(target_matches[pointer:pointer+len(op_val)]).upper()
                    == op_val.upper())
            target_matches[pointer:pointer+len(op_val)] = '-'
            pointer += len(op_val)
        elif op_symbol == '~':
            raise NotImplementedError()
            pass
    
    return target_matches

def compute_read_matches_qual(read_qual, read_info=None, operations=None, 
                      align_start=None, align_end=None, target_length=None,
                      null_val=0):
    """
    Compute which positions in read/query are matched with target during minimap alignment.
    Output phred base quality at the positions that match ('null_val [0]' if no match).
    Based on paf extract operations from cs flag.
    https://lh3.github.io/minimap2/minimap2.html#10

    Parameters
    ----------
    read_seq : str
        read/query aligned sequence.
    operations : list of str
        List of operations provided by cs flag.
        Output of parse_cs_flag().
    align_start : int
        Target alignment start position.
    align_end : int
        Target alignment end position.
    target_length : int
        Target length.

    Raises
    ------
    NotImplementedError
        op_symbol == ~.

    Returns
    -------
    read_matches_qual : ndarray of ints.
        Phred quality scores of read bases that matches the target.

    """
    
    if read_info is not None:
        operations = parse_cs_flag(read_info.cs)
        align_start, align_end, target_length = read_info[['target_start',
                                       'target_end','target_length']].values
    
    read_qual = list(read_qual)
    
    # Start and finish by appeding the missed aligned values -
    read_matches = [null_val] * align_start

    pointer = 0
    for op in operations:
        op_symbol = op[0]
        op_val = op[1:]
        
        if op_symbol == '=': # Identical sequence (long form)
            read_matches += read_qual[pointer:pointer+len(op_val)]
            pointer += len(op_val)
        elif op_symbol == ':': # Identical sequence length (short form)
            read_matches += read_qual[pointer:pointer+int(op_val)]
            pointer += int(op_val)
        elif op_symbol == '*':
            read_matches += [ read_qual[pointer] ]
            pointer += 1
        elif op_symbol == '+':
            pointer += len(op_val)
        elif op_symbol == '-':
            ### Bases in target that are missed, no movement in read/query
            read_matches += [null_val] * len(op_val)
        elif op_symbol == '~':
            raise NotImplementedError()
            pass
    
    read_matches += [null_val] * (target_length - align_end)
    
    return read_matches

def get_qual_per_pos(reads_df, reads_qual, single_read=False):
    """
    Create dataframe with qual score of each position per row.
    Used to transform data for plotting via sns.

    Parameters
    ----------
    reads_df : pd.DataFrame or pd.Series
        DESCRIPTION.
    reads_qual : list of arrays or single array
        DESCRIPTION.
    single_read : bool, optional
        Consider 'reads_df' is a Series and reads_qual a single item.
        The default is False.

    Returns
    -------
    qual_per_pos_df : pd.DataFrame
        DESCRIPTION.

    """
    if single_read:
        reads_df = reads_df.to_frame().T
        reads_qual = [reads_qual]
    
    reads_df['qual_score'] = reads_qual
    reads_df['position'] = reads_df.qual_score.apply(lambda x: list(range(1, len(x)+1)))
    
    # print("Exploding qualities into separate rows...")
    qual_per_pos_df = reads_df.explode(['qual_score','position']).reset_index(drop=True)
    return qual_per_pos_df

def get_ub_area_qual(read_info, read_qual, read_aligned, target_aligned, ub_pos, 
                     margin=5):
    """
    Retrieve quality scores at UB area, given input margin around UB positions.
    Uses read-target alignment to estimate where UB should be located in read.

    Parameters
    ----------
    read_info : pd.Series, row from DataFrame
        DESCRIPTION.
    read_qual : TYPE
        DESCRIPTION.
    read_aligned : TYPE
        Output of compute_alignments().
    target_aligned : TYPE
        Output of compute_alignments().
    ub_pos : TYPE
        DESCRIPTION.
    margin : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    qual_ub_regions : TYPE
        DESCRIPTION.

    """
    ### Create "aligned" qual scores, with np.nan at deletions from ref
    aligned_read_qual = np.full(len(read_aligned), np.nan)
    non_nan_mask = (np.asarray(read_aligned) != '-')
    aligned_read_qual[non_nan_mask] = read_qual

    ### mapping to index, impute empty values with closest position
    mapping_read_qual = np.full(len(read_aligned), np.nan)
    non_nan_mask = (np.asarray(read_aligned) != '-')
    mapping_read_qual[non_nan_mask] = range(len(read_qual))
    ### Replace nans for the nearest/closest
    
    ### Need to replace begin and end, because interpolate only works for inside
    mapping_read_qual[[0,-1]] = np.flatnonzero(non_nan_mask)[[0,-1]]
    #### TODO write code not using pandas to interpolate nearest
    mapping_read_qual = pd.Series(mapping_read_qual).interpolate(method='nearest').values.astype(int)

    map_tar_ali_ref = np.flatnonzero(np.asarray(target_aligned) != '-')
    ali_ub_pos = [ map_tar_ali_ref[p] for p in ub_pos]

    closest_valid_pos = [ mapping_read_qual[p] for p in ali_ub_pos ]
    
    ### Sanity check to see if there is sufficient quals for ub area, align might start/end before
    all_in_read_qual = np.all([ (p-margin > 0 and p+1+margin < len(read_qual)) 
                               for p in closest_valid_pos])
    assert all_in_read_qual # Sanity check ub area inside read_qual
    
    qual_ub_regions = np.asarray([read_qual[p-margin:p+1+margin] 
                                  for p in closest_valid_pos])

    return qual_ub_regions

def get_all_ub_area_qual(paf_df, refs_info, reads_dict, margin=5):
    def generator(input_paf_df, reads_dict, refs_info):
        for read_index, read_info in input_paf_df.iterrows():
            target = refs_info.targets[read_info.target_id]
            x_pos = refs_info.x_pos[read_info.target_id]
            
            ### Load record and store at standard dict
            fake_reads_dict = {read_info.read_id: reads_dict[read_info.read_id]}
            
            input_kwargs = dict(
                margin = margin,
                read_info = read_info,
                reads_dict = fake_reads_dict,
                target = target,
                x_pos = x_pos,
            )
            yield input_kwargs
    
    def get_row_ub_area_qual(input_kwargs):
        read_info = input_kwargs['read_info']
        reads_dict = input_kwargs['reads_dict']
        target = input_kwargs['target']
        x_pos = input_kwargs['x_pos']
        margin = input_kwargs['margin']

        read_qual = get_read_qual(read_info.read_id, None, 
                                  read_info=read_info, reads_dict=reads_dict)
        read_seq = get_read_seq(read_info.read_id, None, 
                                read_info=read_info, reads_dict=reads_dict)
    
        read_aligned, target_aligned = compute_alignments(read_seq, 
                  target, read_info=read_info)
    
        qual_ub_regions = get_ub_area_qual(read_info, read_qual, 
                                           read_aligned, target_aligned, 
                                           x_pos, 
                                           margin=margin)
        return qual_ub_regions.tolist()
    
    print("> Reading UB area quality scores...")
    #### Parallel without progressbar
    # with mp.Pool() as pool:
    #     all_qual_ub_regions = pool.map(get_row_ub_area_qual,
    #               generator(paf_df, reads_dict, refs_info))
    
    #### Parallel with tqdm
    from tqdm.autonotebook import tqdm
    with mp.Pool() as pool:
        all_qual_ub_regions = list(tqdm(
            pool.imap(get_row_ub_area_qual, generator(paf_df, reads_dict, refs_info)), 
            leave=False, desc="Reading UB Q-scores", total=len(paf_df)))
    print()
    
    #### Non-parallel
    # all_qual_ub_regions = [ get_row_ub_area_qual(i)
    #                         for i in generator(paf_df, reads_dict, refs_info)]
    
    return all_qual_ub_regions

def polish_target_matches(target_matches, read_info, target):
    """
    Function to correct minimap2 UB alignment error.
    Examples:
        target : CCCAAYCCCAA
        align  : CGY---CCCAA
        correct: CG---YCCCAA
        
        target : AACAAYAACAA
        align  : GTGG-TYATGA
        correct: GTGGTY-ATGA

    Parameters
    ----------
    target_matches : TYPE
        DESCRIPTION.
    read_info : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    pol_target_matches : TYPE
        DESCRIPTION.

    """
    # ub = ('X' if read_info.strand == '+' else 'Y')
    # if read_info.strand in ['-', 'R']:
    #     target = reverse_complement(target)
    
    ub = 'X' # Always 'X', because target_matches and target are always Forward
    
    pol_target_matches = target_matches.copy()
    ub_positions = [_.start() for _ in re.finditer(ub, target)]
    for ub_pos in ub_positions:
        if target_matches[ub_pos] == ub: # Correct base
            continue
        elif target_matches[ub_pos] == '-': # Del on this pos, check if UB is around
            righttmost = leftmost = ub_pos
            while leftmost > 0 and target_matches[leftmost-1] == '-':
                leftmost -= 1
            while righttmost < len(target_matches)-1 and target_matches[righttmost+1] == '-':
                righttmost += 1
            
            # look for X/Y left to leftmost then right
            if leftmost != 0 and target_matches[leftmost-1] == ub:
                pol_target_matches[leftmost-1] = '-'
                pol_target_matches[ub_pos] = ub
            elif righttmost != len(target_matches)-1 and target_matches[righttmost+1] == ub:
                pol_target_matches[righttmost+1] = '-'
                pol_target_matches[ub_pos] = ub
        elif target_matches[ub_pos-1] == '-' and target_matches[ub_pos+1] == ub: 
            # Example: GTGG- T YATGA
            pol_target_matches[ub_pos-1] = pol_target_matches[ub_pos]
            pol_target_matches[ub_pos] = ub
            pol_target_matches[ub_pos+1] = '-'
        elif target_matches[ub_pos+1] == '-' and target_matches[ub_pos-1] == ub: 
            # Example: GTGGY T -ATGA
            pol_target_matches[ub_pos+1] = pol_target_matches[ub_pos]
            pol_target_matches[ub_pos] = ub
            pol_target_matches[ub_pos-1] = '-'
            # print(''.join(target_matches[ub_pos-5:ub_pos+6]))
    
    return pol_target_matches

def compute_errors_paf(read_info, target, verbose=False, ignore_N=False, 
                       read_seq=None, return_target_matches=False, polish=True):
    operations = parse_cs_flag(read_info.cs)
    
    if read_seq is None:
        target_matches = compute_target_matches(target, operations, 
                                  read_info.target_start, read_info.target_end)
    else:
        target_matches = compute_read_matches(read_seq, operations, 
                      *read_info[['target_start','target_end','target_length']])
    # print(np.asarray([np.asarray(list(target)), target_matches]).T)
        
    ### Polish target_matches to account for minimap2 ub alignment error
    if polish:
        target_matches = polish_target_matches(target_matches, read_info, target)
    
    # errors = (np.asarray(list(target)) != target_matches).astype(int)
    errors = (np.asarray(list(target)) != target_matches).astype(float)
        
    if ignore_N:
        n_positions = [_.start() for _ in re.finditer('N', target)]
        errors[n_positions] = 0
    
    #### TODO ignore errors before start and after end
    # ignore_missalign = True
    # if ignore_missalign:
    #     errors[:read_info.target_start] = np.nan
    #     errors[read_info.target_end:] = np.nan
    
    if verbose:
        print("Expected n_matches: ", read_info.n_matches)
        print("Estimated n_matches:", len(errors)-errors.sum())

        print("Positions with errors:")
        print(np.asarray([np.asarray(list(target))[np.argwhere(errors).flat[:]],
                          np.argwhere(errors).flat[:]]).T)
    
    if read_info.strand in ['-', 'R']:
        errors = errors[::-1]
    
    if not return_target_matches:
        return errors
    else:
        return errors, target_matches

def compute_error_rate_per_pos_paf(paf_df, target, ignore_N=False, 
                                   reads_dict=None, targets=None):
    
    if targets is not None:
        target = list(targets.values())[0]
    
    if paf_df.shape[0] == 0:
        # error_rate = np.full(len(target), np.nan)
        error_rate = np.full(len(target), 100.)
        return error_rate, [], {}
    
    all_n_matches = []
    all_ub_area_metrics = []
    all_errors = []
    
    def my_gen(reads_info_df, reads_dict, target, targets):
        for read_index, read_info in reads_info_df.iterrows():
            if targets is not None:
                target = targets[read_info.target_id]
                
                if (reads_dict is not None and 
                        not read_info.target_id.startswith('PC')):
                    target = target.replace('N', 'X')
            
            read_seq = (None if reads_dict is None else 
                        get_read_seq(read_info.read_id, None, 
                                     read_info=read_info, 
                                     reads_dict=reads_dict))
            
            yield read_info, read_seq, target
    
    def _compute_errors_and_ub_metrics(input_tuple):
        read_info, read_seq, target = input_tuple
        
        errors, target_matches = compute_errors_paf(read_info, target, 
                ignore_N=ignore_N, read_seq=read_seq, return_target_matches=True)
        
        #### [OPT] Ignore errors before start and after end
        ignore_edges = False
        if ignore_edges:
            errors = np.asarray(errors, dtype=float)
            errors[:read_info.target_start] = np.nan
            errors[read_info.target_end:] = np.nan
        
        n_matches = len(errors) - errors.sum()
        
        if read_seq is not None :
            #### get_ub_area_metrics
            kmer_len = 6
            x_positions = [_.start() for _ in re.finditer('[NXY]', target)]
            ub_area_mask = np.full(len(target), False)
            for x_pos in x_positions:
                x_pos_slice = slice(x_pos + 1 - kmer_len, x_pos + kmer_len)
                ub_area_mask[x_pos_slice] = True
            ub_area_mask[x_positions] = False
            
            ### target_matches is always Forward
            inclusive_ub_area_mask = ub_area_mask.copy()
            inclusive_ub_area_mask[x_positions] = True
            ub_area_seq = np.asarray(target_matches)[inclusive_ub_area_mask]
            ub_area_seq = ''.join(ub_area_seq)
            tar_align_slice = slice(read_info.target_start, read_info.target_end)
            tar_align_mask = np.full(len(target), False)
            tar_align_mask[tar_align_slice] = True
            
            if read_info.strand in ['R','-']:
                ub_area_seq = reverse_complement(ub_area_seq)
                ub_area_mask = np.flip(ub_area_mask)
                inclusive_ub_area_mask = np.flip(inclusive_ub_area_mask)
                tar_align_mask = np.flip(tar_align_mask)
                x_positions= [ len(target) - p - 1 for p in x_positions[::-1] ]
            
            ub_area_matches = np.logical_not(errors[ub_area_mask]).sum()
            ub_area_len = ub_area_mask.sum()
            
            ub_matches = np.logical_not(errors[x_positions]).sum()
            ub_len = len(x_positions)
            
            #### false_discovery_rate
            ubs_detected = np.isin(target_matches, ['X','Y']).sum()
            # ubs_detected_ub_A = np.isin(target_matches[ub_area_mask], ['X','Y']).sum()
            # ubs_detected_non_ub_A = np.isin(target_matches[~ub_area_mask], ['X','Y']).sum()
            false_ubs_detected = ubs_detected - ub_matches
            if ubs_detected > 0:
                false_discovery_rate = false_ubs_detected/ubs_detected
            else:
                false_discovery_rate = np.nan
            # print(false_discovery_rate, false_ubs_detected, ubs_detected, ub_matches)
            false_positive_rate = false_ubs_detected/(len(target) - ub_len)
            
            non_ub_area_mask = np.logical_not(inclusive_ub_area_mask)
            non_ub_area_matches = np.logical_not(errors[non_ub_area_mask]).sum()
            non_ub_area_len = non_ub_area_mask.sum()
            
            #### Metrics per x_pos
            ub_acc_per_pos, ub_area_acc_per_pos, ub_area_acc_plus_per_pos = [],[],[]
            for x_pos in x_positions:
                ub_matches_pos = np.logical_not(errors[x_pos]).sum()
                
                x_pos_slice = slice(x_pos + 1 - kmer_len, x_pos + kmer_len)
                ub_area_matches_pos = np.logical_not(errors[x_pos_slice]).sum() - ub_matches_pos
                
                ub_acc_per_pos.append( ub_matches_pos/1 )
                ub_area_acc_per_pos.append( ub_area_matches_pos/10 )
                ub_area_acc_plus_per_pos.append( (ub_area_matches_pos+ub_matches_pos)/11 )
            
            tar_align_len = read_info.target_end - read_info.target_start
            tar_alig_n_matches = int(tar_align_len - errors[tar_align_mask].sum())
            
            if ub_len > 0: 
                ub_area_acc = ub_area_matches/ub_area_len
                ub_acc = ub_matches/ub_len
                ub_area_acc_plus = (ub_area_matches+ub_matches)/(ub_area_len+ub_len)
            else: # Probably PC, not XNA
                ub_area_acc = ub_acc = ub_area_acc_plus = np.nan
            
            ub_area_metrics = dict(
                ub_area_acc = ub_area_acc,
                ub_area_matches = int(ub_area_matches),
                ub_area_len = int(ub_area_len),
                ub_area_seq = ub_area_seq,
                
                ub_acc = ub_acc,
                ub_matches = int(ub_matches),
                ub_len = int(ub_len),
                ub_area_acc_plus = ub_area_acc_plus,
                
                non_ub_area_acc = non_ub_area_matches/non_ub_area_len,
                non_ub_area_matches = int(non_ub_area_matches),
                non_ub_area_len = int(non_ub_area_len),
                
                ub_acc_per_pos = ub_acc_per_pos,
                ub_area_acc_per_pos = ub_area_acc_per_pos,
                ub_area_acc_plus_per_pos = ub_area_acc_plus_per_pos,
                label_per_pos = x_positions,
                
                target_alig_acc = tar_alig_n_matches/tar_align_len,
                
                fdr = false_discovery_rate,
                fpr = false_positive_rate,
                
                true_pos = int(ub_matches), # UBs called as UBs
                false_neg = int(ub_len - ub_matches), # UBs called wrongly
                true_neg = int(len(target) - ub_len - false_ubs_detected), # Nat bases called right
                false_pos = int(false_ubs_detected),  # Nat bases called as UBs
                
                #### TODO Add or not index value here for accurate attribution later on? didnt worked
                # read_index = read_info.name, 
            )
        else:
            ub_area_metrics = {}
        
        return errors, n_matches, ub_area_metrics
    
    if targets is None:
        with mp.Pool() as pool:
            # all_errors, all_n_matches, all_ub_area_metrics = pool.map(
            output_tuples = pool.map(
                _compute_errors_and_ub_metrics,
                my_gen(paf_df, reads_dict, target, targets))
    else:
        from tqdm.autonotebook import tqdm
        with mp.Pool() as pool:
            output_tuples = list(tqdm(
                pool.imap(_compute_errors_and_ub_metrics,
                          my_gen(paf_df, reads_dict, target, targets)), 
                leave=False, desc="Computing error rates per position", total=len(paf_df)))
        print()
    
    all_errors = [ o[0] for o in output_tuples ]
    all_n_matches = [ o[1] for o in output_tuples ]
    all_ub_area_metrics = [ o[2] for o in output_tuples ]
    
    ### Transforming list of dicts into dict of lists
    if reads_dict is not None :
        all_ub_area_metrics = {k: [dic[k] for dic in all_ub_area_metrics] 
                           for k in all_ub_area_metrics[0]}
    else:
        all_ub_area_metrics = {}
    
    if targets is None:
        error_rate = np.nanmean(all_errors, axis=0)*100
    else: # Means multiple targets, so a single error_rate array makes no sense
        # error_rate = None
        ### Create dict with str-tar (to be used by compute_all_error_rates_paf)
        # Iter grp by and get indices to be used for all_errors
        error_rate = {}
        all_errors = np.asarray(all_errors, dtype=object)
        for tar_str, group in paf_df.reset_index().groupby(['target_id','strand']):
            tar_str_errors = all_errors[group.index]
            if len(tar_str_errors.shape) == 1: # When all_errors is ragged nested sequences 
                tar_str_errors = np.stack(tar_str_errors)
            error_rate[tar_str] = np.nanmean(tar_str_errors, axis=0)*100
    
    return error_rate, all_n_matches, all_ub_area_metrics

def add_ub_area_metrics(paf_df, targets, reads_dict=None):
    """
    *Important, pad_df is appended here.
    > read_acc, target_acc and ub_area_metrics (ub_area_: acc, matches, seq, len)

    Parameters
    ----------
    paf_df : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    reads_dict : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    paf_df : TYPE
        DESCRIPTION.

    """
    
    #### TODO change this to another func: 
    ### a) no need for error_rate; b) fix index dependence issue
    error_rate, all_n_matches, all_ub_area_metrics = \
        compute_error_rate_per_pos_paf(paf_df, target=None, ignore_N=False, 
                                       reads_dict=reads_dict, targets=targets)
    
    # print(all_ub_area_metrics['read_index'])
    # read_indexes = all_ub_area_metrics.pop('read_index')
    read_indexes = paf_df.index
    # print(read_indexes)
    
    paf_df.loc[read_indexes,'read_acc'] = all_n_matches/paf_df.read_alignment_length
    paf_df.loc[read_indexes,'target_acc'] = all_n_matches/paf_df.target_length
    
    
    for key, val in all_ub_area_metrics.items():
        # print(paf_df.index, key)
        if type(val) is not list:
            paf_df.loc[read_indexes, key] = val
        else:
            # print(paf_df.index, key, val)
            paf_df.loc[read_indexes, key] = pd.Series(val)
    
    return paf_df

def complement_paf(paf_df, reads_filepath=None, reads_dict=None,
                   add_ub_acc=False, add_bc=False, ref_info=None,
                   max_barcode_dist=None):
    if ref_info is None:
        from misc.xna_refs import identify_ref
        ref_info = identify_ref(paf_df.target_id.unique())

    if 'barcode_distance' not in paf_df:
        print("Computing barcode_distance...")
        if reads_dict is None:
            reads_dict = index_reads_file(reads_filepath)
        # paf_df = add_barcode_info(paf_df, ref_info, reads_dict, inplace=True, parallel=True)
        add_barcode_info(paf_df, ref_info, reads_dict, inplace=True, parallel=True)
    
    if max_barcode_dist is not None:
        paf_df = filter_demux(paf_df, max_barcode_dist=max_barcode_dist)
        paf_df.reset_index(drop=True, inplace=True)
    
    if 'ub_area_seq' not in paf_df:
        print("Computing ub area stats...")
        if reads_dict is None:
            reads_dict = index_reads_file(reads_filepath)
        add_ub_area_metrics(paf_df, ref_info.targets, reads_dict=reads_dict)

    return paf_df

def UNTESTED_filter_paf(paf_df, max_bc_dist, ref_info): ### Probably can just use filter demux?
    if ref_info.barcode_len == 24:
        max_bc_dist = 5 # 5? 6?
    elif ref_info.barcode_len == 30:
        max_bc_dist = 8 # ?6
    print("Filtering read by max barcode distance:", max_bc_dist)
    filtered_paf_df = paf_df[paf_df.barcode_distance <= max_bc_dist]

    print(paf_df.shape[0])
    print(filtered_paf_df.shape[0])

    filter_non_ub_coverage= False
    if filter_non_ub_coverage:
        print("[WARNING] Filtering reads which do not cover ub region.")
        valid_mask = filtered_paf_df.apply(lambda row: 
                     (row.target_start < ref_info.xna_kmers_pos[row.target_id][0] and
                      row.target_end > ref_info.xna_kmers_pos[row.target_id][1]), 
                     axis=1)
        
        filtered_paf_df = filtered_paf_df[valid_mask]  
        
        print(filtered_paf_df.shape[0])

    print(paf_df.groupby(['strand'])[['ub_acc','ub_area_acc','ub_area_acc_plus']].mean())
    print(filtered_paf_df.groupby(['strand'])[['ub_acc','ub_area_acc','ub_area_acc_plus']].mean())

    # sorted(filtered_paf_df.target_id.unique())
    # filtered_paf_df.sort_values('barcode_distance')[['target_id','strand','barcode_distance']]
    return filtered_paf_df

### Non-parallel version compute_all_error_rates_paf()
def DEPREC_compute_error_rate_per_pos_paf(paf_df, target, ignore_N=False, reads_dict=None,
                                   targets=None):
    all_n_matches = []
    all_ub_area_metrics = []
    
    if targets is not None:
        target = list(targets.values())[0]
    
    if paf_df.shape[0] == 0:
        # error_rate = np.full(len(target), np.nan)
        error_rate = np.full(len(target), 100.)
        return error_rate, all_n_matches, {}
    
    # errors_sum = np.zeros(len(target), dtype=int)
    errors_sum = np.zeros(len(target), dtype=float)
    all_errors = []
    
    #### TODO parallel version?
    for read_index, read_info in paf_df.iterrows():
        if targets is not None:
            target = targets[read_info.target_id]
        
        #### TODO remove get_read_seq?
        read_seq = (None if reads_dict is None else 
                    get_read_seq(read_info.read_id, None, 
                                 read_info=read_info, reads_dict=reads_dict))
        
        # errors = compute_errors_paf(read_info, target, ignore_N=ignore_N, read_seq=read_seq)
        errors, target_matches = compute_errors_paf(read_info, target, 
                ignore_N=ignore_N, read_seq=read_seq, return_target_matches=True)
        ### This sanity check works for PC, but not for XNA with xna model.
        ### Because minimap target_cover skips UB(N) position, which might be correct.
        # assert np.isclose(read_info.target_cover, 1-errors.sum()/len(errors))
        
        #### [OPT] Ignore errors before start and after end
        ignore_edges = False
        if ignore_edges:
            errors = np.asarray(errors, dtype=float)
            errors[:read_info.target_start] = np.nan
            errors[read_info.target_end:] = np.nan
            ### Invert if reverse? R: no need, it is the same? Looks strange
            # if read_info.strand == '+':
            #     errors[:read_info.target_start] = np.nan
            #     errors[read_info.target_end:] = np.nan
            # else:
            #     errors[read_info.target_start:] = np.nan
            #     errors[:read_info.target_end] = np.nan
        
        errors_sum += errors
        all_errors.append(errors)
        
        if reads_dict is not None :
            n_matches = len(errors) - errors.sum()
            all_n_matches.append(n_matches)
            
            #### get_ub_area_metrics
            kmer_len = 6
            x_positions = [_.start() for _ in re.finditer('[NXY]', target)]
            ub_area_mask = np.array([False]*len(target))
            for x_pos in x_positions:
                x_pos_slice = slice(x_pos + 1 - kmer_len, x_pos + kmer_len)
                ub_area_mask[x_pos_slice] = True
                ub_area_mask[x_pos] = False
            
            ### target_matches is always Forward
            inclusive_ub_area_mask = ub_area_mask.copy()
            inclusive_ub_area_mask[x_positions] = True
            ub_area_seq = np.asarray(target_matches)[inclusive_ub_area_mask]
            ub_area_seq = ''.join(ub_area_seq)
            
            if read_info.strand in ['R','-']:
                ub_area_seq = reverse_complement(ub_area_seq)
                ub_area_mask = np.flip(ub_area_mask)
                x_positions= [ len(target) - p - 1 for p in x_positions[::-1] ]
            
            ub_area_matches = np.logical_not(errors[ub_area_mask]).sum()
            ub_area_len = ub_area_mask.sum()
            
            ub_matches = np.logical_not(errors[x_positions]).sum()
            ub_len = len(x_positions)
            
            non_ub_area_mask = np.logical_not(inclusive_ub_area_mask)
            non_ub_area_matches = np.logical_not(errors[non_ub_area_mask]).sum()
            non_ub_area_len = non_ub_area_mask.sum()
            
            ub_area_metrics = dict(
                ub_area_acc = ub_area_matches/ub_area_len,
                ub_area_matches = int(ub_area_matches),
                ub_area_len = int(ub_area_len),
                ub_area_seq = ub_area_seq,
                
                ub_acc = ub_matches/ub_len,
                ub_matches = int(ub_matches),
                ub_len = int(ub_len),
                ub_area_acc_plus = (ub_area_matches+ub_matches)/(ub_area_len+ub_len),
                
                non_ub_area_acc = non_ub_area_matches/non_ub_area_len,
                non_ub_area_matches = int(non_ub_area_matches),
                non_ub_area_len = int(non_ub_area_len),
            )
            
            all_ub_area_metrics.append(ub_area_metrics)
            
    ### Transforming list of dicts into dict of lists
    if reads_dict is not None :
        all_ub_area_metrics = {k: [dic[k] for dic in all_ub_area_metrics] 
                           for k in all_ub_area_metrics[0]}
    else:
        all_ub_area_metrics = {}
    
    # error_rate = (errors_sum*100)/paf_df.shape[0]
    error_rate = np.nanmean(all_errors, axis=0)*100
    
    # print(np.nansum(all_errors, axis=0))
    # print(np.shape(all_errors))
    # print(np.count_nonzero(np.isnan(all_errors), axis=0))
    # print(read_info.target_id, read_info.strand, np.count_nonzero(np.isnan(all_errors)))
    # print(error_rate[:10], error_rate[-10:])
    # print(error_rate2)
    # print(error_rate[:5])
    # print(error_rate2[:5])
    # print(np.allclose(error_rate, error_rate2))
    
    # return error_rate
    # return error_rate, all_n_matches
    return error_rate, all_n_matches, all_ub_area_metrics

def compute_all_error_rates_paf(paf_df, targets, sample_size=None, reads_dict=None):
    """
    Compute error rate for each position, per target.
    Ex: target with 89 bases, compute error rate for each of it. 
        Return dict with key =(target_id, strand) and val=array with 89 error rates.
    
    *Important, pad_df is appended here.
    > read_acc, target_acc and ub_area_metrics (ub_area_: acc, matches, seq, len)

    Parameters
    ----------
    paf_df : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    sample_size : TYPE, optional
        DESCRIPTION. The default is None.
    reads_dict : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    error_rates_dict : TYPE
        DESCRIPTION.
        {('XNA03', 'R'): [ 79.12119064  67.28561304  17.88093551 ...]
         ('XNA03', 'F'): [ 79.12119064  67.28561304  17.88093551 ...]}

    """
    ### Implement with multithreading? R: Done, inside compute_error_rate_per_pos_paf()
    # Too slow if using all the data
    # Sample size is interesting workaround, 
    # and in fact I should be running this for validation data only (less data)
    # *But might still be an issue when running for XNA 1024
    if sample_size is not None:
        print("[Warning] Computing error rates using sample size of:", sample_size)
    
    if paf_df.target_id.nunique() != len(targets):
        pass
        # print("[Warning] Mismatch between aligned number of targets and reference total:",
        #       paf_df.target_id.nunique(), "out of", len(targets))
    
    if True:
        ### Drop loop tar-str, call entire paf
        # How to split error_rate per tar-str? Needs to adapt the other func
        error_rates_dict, all_n_matches, all_ub_area_metrics = compute_error_rate_per_pos_paf(
            paf_df, target=None, ignore_N=False, reads_dict=reads_dict, 
            targets=targets)
        #### [WARNING] updating paf_df object
        if reads_dict is not None:
            read_indexes = paf_df.index
            paf_df.loc[read_indexes,'read_acc'] = all_n_matches/paf_df.read_alignment_length
            paf_df.loc[read_indexes,'target_acc'] = all_n_matches/paf_df.target_length
            for key, val in all_ub_area_metrics.items():
                if type(val) is not list:
                    paf_df.loc[read_indexes, key] = val
                else:
                    paf_df.loc[read_indexes, key] = pd.Series(val, index=read_indexes)
    else: # Overly-complicated loop
        error_rates_dict = {}
        # for target_id in paf_df.target_id.unique(): # Compute only for those in paf
        for target_id in targets.keys(): # Compute for all target ids
            # target = ref_info.targets[target_id]
            target = targets[target_id]
            if reads_dict is not None and not target_id.startswith('PC'):
                target = target.replace('N', 'X')
            # tar_type = ('PC' if target_id.startswith('PC') else 'XNA')
            for strand in ['F','R']:
                # print(target_id, strand)
                strand_paf = ('+' if strand == 'F' else '-')
                paf_mask = (( paf_df.target_id == target_id) & 
                            ( paf_df.strand.isin([strand, strand_paf]) ) )
                tar_str_paf_df = paf_df[paf_mask]
                
                if tar_str_paf_df.shape[0] == 0:
                    # print("[Warning] no alignment for:", target_id, strand)
                    continue
                
                if sample_size is not None:
                    tar_str_paf_df = tar_str_paf_df.head(sample_size)
                
                ### TODO Create argument to get_ub_area_metrics here or not?
                # error_rate, all_n_matches, all_ub_area_metrics = DEPREC_compute_error_rate_per_pos_paf(
                #     tar_str_paf_df, target, ignore_N=False, reads_dict=reads_dict)
                error_rate, all_n_matches, all_ub_area_metrics = compute_error_rate_per_pos_paf(
                    tar_str_paf_df, target, ignore_N=False, reads_dict=reads_dict)
                
                # error_rates_dict[(target_id,strand,tar_type)] = error_rate
                error_rates_dict[(target_id, strand)] = error_rate
                
                #### [WARNING] updating paf_df object
                if reads_dict is not None:
                    # print(f"{xxx=}")
                    paf_df.loc[tar_str_paf_df.index,'read_acc'] = (
                        all_n_matches/tar_str_paf_df.read_alignment_length)
                    paf_df.loc[tar_str_paf_df.index,'target_acc'] = (
                        all_n_matches/tar_str_paf_df.target_length)
                    
                    for key, val in all_ub_area_metrics.items():
                        if type(val) is not list:
                            paf_df.loc[tar_str_paf_df.index, key] = val
                        else:
                            paf_df.loc[tar_str_paf_df.index, key] = pd.Series(val, 
                                                      index=tar_str_paf_df.index)
    
    return error_rates_dict

def compute_errors(target, query, verbose=True, ignore_N=False):
    scores_kwargs = dict(
        ### minimap scores and penalties
        match_score = 2, mismatch_score = -4,
        target_open_gap_score = -4.0, query_open_gap_score = -24.0,
        target_end_gap_score = -2.0, query_end_gap_score = -1.0,
    )
    aligner = Align.PairwiseAligner(mode='global', **scores_kwargs)
    
    alignments = aligner.align(target, query)
    
    if verbose:
        print("Number of alignments generated:", len(alignments))
    
    errors = np.ones(len(target), dtype=int)
    
    if verbose:
    # if True:
        print("Warning: Using first alignment out of", len(alignments))
    alignment = alignments[0]
    
    for tar_range, qry_range in np.swapaxes(alignment.aligned, 0,1):
        tar_slice = slice(*tar_range)
        qry_slice = slice(*qry_range)
        
        tar_str = np.array(list(target[tar_slice]))
        qry_str = np.array(list(query[qry_slice]))
        
        # Looking for mismatches
        errors[tar_slice] = (tar_str != qry_str).astype(int)
        # Slices not updated means they were not aligned, therefore not correct
    
    if ignore_N:
        n_positions = [_.start() for _ in re.finditer('N', target)]
        errors[n_positions] = 0
    
    return errors

def compute_error_rate_per_pos(sample_reads, read_seqs, target, ignore_N=False):
    count = 0
    errors_sum = np.zeros(len(target), dtype=int)
    for read_id, read_info in sample_reads.iterrows():
        if read_id not in read_seqs:
            continue
        count += 1
        read_seq = read_seqs[read_id]
        if not isinstance(read_seq, str):
            read_seq = str(read_seq.seq)
    # for read_id, read_seq in read_seqs.items():
    #     read_info = sample_reads.loc[read_id]
        # read_seq = read_seq[read_info.read_start:read_info.read_end+1]
        read_seq = read_seq[read_info.read_start:read_info.read_end]
        
        errors = compute_errors(target, read_seq, verbose=False, ignore_N=ignore_N)
        errors_sum += errors
    
    # error_rate = (errors_sum*100)/sample_reads.shape[0]
    # error_rate = (errors_sum*100)/len(read_seqs)
    error_rate = (errors_sum*100)/count
    
    return error_rate

def add_read_location(reads_info, refs_info):
    if isinstance(reads_info, pd.DataFrame):
        res = reads_info.apply(lambda x: refs_info.locate_read(
            x.barcode_start, x.barcode_end, x.target_id, x.strand, x.read_length), 
            axis=1)
        
        with pd.option_context('mode.chained_assignment', None):
            reads_info[['read_start', 'read_end']] = pd.DataFrame(res.tolist(), 
                                                            index=res.index)
    else: # Assuming it is pd.Series
        read_start, read_end = refs_info.locate_read(reads_info.barcode_start, 
             reads_info.barcode_end, reads_info.target_id,
             reads_info.strand, reads_info.read_length)
        
        with pd.option_context('mode.chained_assignment', None):
            reads_info['read_start'] = read_start
            reads_info['read_end'] = read_end
        
    
    return reads_info

def get_barcode_match_score(read_info, read_seq, left_primer, barcode,
                            n_relax_bases=3):
    
    # if is_forward:
    if read_info.strand == '+': # is_forward
        read_start = read_info['read_start']
        read = read_seq
    else:
        read = reverse_complement(read_seq)
        read_start = len(read)-read_info['read_end']
    
    # Extract barcode region
    # mapping contain barcode region
    if left_primer >= read_info['target_start']:        
        start = left_primer - read_info['target_start'] + read_start
    # mapping contain partial/none of barcode region
    else:
        start = np.max([read_start - (read_info['target_start'] - left_primer), 0])

    # Find exact barcode region
    best_score = np.inf
    best_start = None
    best_end = None
    best_obs_bc = None
    
    # target_seq = targets[read_info['target_id']]
    
    bc_length = len(barcode)
    for i in range(np.max([start-n_relax_bases, 0]), start+n_relax_bases+1):
        j = i + bc_length
        obs_bc = read[i:j]
        
        score = Levenshtein.distance(barcode, obs_bc)
        if score < best_score:
            best_score = score
            best_start = i
            best_end = j
            best_obs_bc = obs_bc

    best_bc_info = dict(
        barcode_detected = best_obs_bc,
        barcode_detected_len = len(best_obs_bc),
        barcode_start = best_start,
        barcode_end = best_end,
        barcode_distance = best_score,
    )

    return best_bc_info

def add_barcode_info(orig_paf_df, ref_info, reads_dict, n_relax_bases = 3,
                     inplace=False, parallel=False):
    paf_df = orig_paf_df
    bc_cnt_map = {bc: cnt for bc, cnt in 
         zip(*np.unique(list(ref_info.barcodes.values()), return_counts=True))}
    
    if parallel:
        def my_gen(orig_paf_df, reads_dict, ref_info):
            for read_index, read_info in orig_paf_df.iterrows():
                read_seq = str(reads_dict[read_info.read_id].seq)
                left_primer_len = ref_info.left_primer_len
                barcode = ref_info.barcodes[read_info.target_id]
                yield read_info, read_seq, left_primer_len, barcode
        
        def _compute_barcode(input_tuple):
            read_info, read_seq, left_primer_len, barcode = input_tuple
            
            best_bc_info = {}
            best_bc_info['index'] = read_info.name
            best_bc_info['barcode'] = barcode
            best_bc_info['barcode_cnt'] = bc_cnt_map[barcode]
            
            best_bc_info_ret = get_barcode_match_score(read_info, read_seq, 
                    left_primer_len, barcode, n_relax_bases=n_relax_bases)
            
            for k, v in best_bc_info_ret.items():
                best_bc_info[k] = v
            
            return best_bc_info
            # return read_info.name, best_bc_info
        
        # print("> Computing barcodes...")
        with mp.Pool() as pool:
            best_bc_info_list = pool.map(_compute_barcode,
                     my_gen(orig_paf_df, reads_dict, ref_info))
        
        # print("> Generating barcodes df...")
        best_bc_info_dict = {k: [dic[k] for dic in best_bc_info_list] 
                            for k in best_bc_info_list[0]}
        best_bc_info_df = pd.DataFrame(best_bc_info_dict).set_index('index')
        # print("> Adding barcodes to paf...")
        if inplace:
            # print(">> inplace...")
            # print(best_bc_info_df)
            # for col_name, col_vals in best_bc_info_df.iteritems():
            for col_name, col_vals in best_bc_info_df.items():
                # print(col_name)
                # print(col_vals)
                paf_df.loc[best_bc_info_df.index, col_name] = col_vals
                # raise NotImplementedError()
        else:
            # print(">> copy...")
            paf_df = paf_df.merge(best_bc_info_df, left_index=True, right_index=True)
    else:
        for read_index, read_info in orig_paf_df.iterrows():
            read_seq = str(reads_dict[read_info.read_id].seq)
            
            barcode = ref_info.barcodes[read_info.target_id]
            
            best_bc_info = get_barcode_match_score(read_info, read_seq, 
                    ref_info.left_primer_len, barcode, n_relax_bases=n_relax_bases)
            
            paf_df.loc[read_index, 'barcode'] = barcode # barcode_reference
            
            for k,v in best_bc_info.items():
                paf_df.loc[read_index, k] = v
    
    return paf_df

def print_reads_stats(reads_df, ref_info=None):
    if ref_info is not None:
        exp_len = ref_info.long_targets_len[0]
        tpl_len = ref_info.targets_len[0]
        left_arm_len = ref_info.left_arm_len
        right_arm_len = ref_info.right_arm_len
    else:
        # exp_len = tpl_len = left_arm_len = right_arm_len = 'N/A'
        exp_len = tpl_len = left_arm_len = right_arm_len = np.nan
    
    print("Expected read length:", exp_len)
    # with pd.option_context("display.float_format", '{:.2f}'.format):
        # print(reads_df[['target_cover','read_length']].describe())
        # print(reads_df[['target_cover','read_length','n_matches',
        #         'target_length','block_length', 'percent_match',
        #         'target_start', 'target_end']].describe().T)
    with pd.option_context("display.float_format", '{:.0f}'.format):
        print(reads_df[['read_length']].describe().drop('count').T)
    with pd.option_context("display.float_format", '{:.0%}'.format):
        print(reads_df[['template_coverage','percent_match']].describe().drop('count').T)
    
    if 'ub_area_acc' in reads_df:
        with pd.option_context("display.float_format", '{:.0%}'.format):
            print(reads_df[['ub_area_acc','target_acc','read_acc']].describe().drop('count').T)
    
    print("Expected read start and end:")
    with pd.option_context("display.float_format", '{:.0f}'.format):
        # print(filter_paf_df[['read_start','read_end']].describe())
        print("\tF:", left_arm_len, left_arm_len + tpl_len)
        print(reads_df.loc[reads_df.strand=='F', 
                           ['read_start','read_end']].describe().T)
        print("\tR:", right_arm_len, right_arm_len + tpl_len)
        print(reads_df.loc[reads_df.strand=='R', 
                           ['read_start','read_end']].describe().T)
    
    
    print("Read count per tar+str statistics:")
    with pd.option_context("display.float_format", '{:.0f}'.format):
        print(reads_df.groupby(['target_id', 'strand']).size().describe().to_frame().T)
        # print(reads_df.groupby(['target_id', 'strand']).size())

def get_tar_reads_count(reads_df, targets_id, print_stats=False, agg_min_strands=True):
    # TODO targets_id=None? Then do not complement with all_target
    if not isinstance(targets_id, list):
        targets_id = list(targets_id)
    
    print("Counting reads grouped by target id and strand...")
    target_read_count_df = reads_df.groupby(['target_id', 'strand']).size().rename('n_read')
    target_read_count_df = target_read_count_df.reset_index()
    with pd.option_context("display.float_format", '{:,.0f}'.format):
        p = [.01,.05,.10,.25,.75]
        print(pd.concat((
            target_read_count_df.groupby('strand').n_read.describe(percentiles=p),
            target_read_count_df[['n_read']].describe(percentiles=p).T.rename({'n_read':'FR'}),
            )) )
        print(target_read_count_df.groupby('strand').n_read.sum().astype(float).to_frame().T)
    # print("head:", target_read_count_df.n_read.sort_values().head(16).to_frame().T.values)
    # print("tail:", target_read_count_df.n_read.sort_values().tail(10).to_frame().T.values)
    
    ### Sorting by n_read
    pivot_read_cnt_df = target_read_count_df.pivot(
        index='target_id', columns='strand', values='n_read').fillna(0)
    pivot_read_cnt_df['min'] = pivot_read_cnt_df.min(axis=1)
    pivot_read_cnt_df['sum'] = pivot_read_cnt_df[['R','F']].sum(axis=1)
    pivot_read_cnt_df = pivot_read_cnt_df.sort_values(['min','sum']).drop(columns=['min','sum'])
    with pd.option_context("display.float_format", '{:,.0f}'.format):
        print(pivot_read_cnt_df.head(16).T)
        print(pivot_read_cnt_df.tail(16).T)
    
    all_target_read_count_df = pd.DataFrame(
        {'target_id': 2*targets_id, 
         'strand': len(targets_id)*['F'] + len(targets_id)*['R'], 
         'n_read': 2*len(targets_id)*[0]})
    
    all_target_read_count_df.set_index(['target_id','strand'], inplace=True)
    target_read_count_df.set_index(['target_id','strand'], inplace=True)
    all_target_read_count_df.update(target_read_count_df)
    
    # .update() changes dtype to float (pandas known issue), recasting to int
    all_target_read_count_df = all_target_read_count_df.astype(int)
    all_target_read_count_df.reset_index(inplace=True)
    all_target_read_count_df = all_target_read_count_df.sort_values(
        ["target_id","n_read"])
    
    if agg_min_strands:
        min_all_target_read_count_df = all_target_read_count_df.drop_duplicates(
            'target_id', ignore_index=True)
    else:
        min_all_target_read_count_df = all_target_read_count_df
    
    #### Printing num reads bins
    if print_stats:
        # print("Statistics over number of reads per template:")
        # print(min_all_target_read_count_df.groupby('strand').n_read.describe().T)
        
        max_n_read = min_all_target_read_count_df.n_read.max() + 1
        if max_n_read <= 251:
            max_n_read = 276
        # bins = [-1] + list(range(0,251,25)) + [max_n_read]
        #### Hard-coded smaller binarization (finer)
        bins = [0,1,5,10,15,20,25,30,35,40,45,50,55,60] + [max_n_read]
        # bins = [0,1,5,10,15,20,25,30,35] + [max_n_read]
        
        binned_counts = pd.cut(min_all_target_read_count_df.n_read, bins,
                               right=False).value_counts()
        # binned_counts = pd.cut(min_all_target_read_count_df.n_read, bins, 
        #                        right=False).value_counts()
        binned_counts.sort_index(inplace=True)
        
        print("Number of templates binned by number of reads (min between strands):")
        # print(binned_counts)
        binned_counts_df = pd.DataFrame(
            {'count':binned_counts, 
             'perc': binned_counts/binned_counts.sum(),
             'sum': binned_counts.cumsum(),
             'sum (%)': binned_counts.cumsum()/binned_counts.sum()})
        binned_counts_df.index.rename('bins', inplace= True)
        with pd.option_context("display.float_format", '{:.2%}'.format):
            print(binned_counts_df)
        
        print("Lowest read counts:",
              min_all_target_read_count_df.n_read.sort_values().values[:10])
    
    return min_all_target_read_count_df

def slice_eventalign(eventalign_df, refs_info, target_id, strand, 
                     kmer_len=6, use_pc_majority=True, margin=0, reverse_eventalign=False):
    
    #### TODO do it iteratively, and add label to which ub it is (for multi-UBs case), 
    ### So that I can use this information later
    
    ### Changed the focus logic to be ub pos oriented, to avoid adding middle positions
    xna_target_id = (target_id if not target_id.startswith('PC') else 
                     refs_info.get_complement_target_id(target_id))
    focus_pos = []
    for p in refs_info.x_pos[xna_target_id]:
        # focus_pos += list(range(p-kmer_len + 1,p+1))
        focus_pos += list(range(p-kmer_len + 1-margin,p+1+margin))
    focus_pos = list(set(focus_pos))
    
    # if margin is not None and margin > 0:
    #     start_focus_pos, end_focus_pos = refs_info.xna_kmers_pos[target_id]
    #     start_focus_pos -= margin - 1 + kmer_len//2
    #     end_focus_pos += margin - 1 - kmer_len//2
    #     focus_pos += list(range(end_focus_pos, end_focus_pos))
    
    # print("Region of focus:", (start_focus_pos,end_focus_pos))
    # print("Number of positions focused:", end_focus_pos-start_focus_pos+1)
    
    sliced_eventalign_df = eventalign_df[eventalign_df.position.isin(focus_pos)]
    
    if reverse_eventalign:
        target = reverse_complement(refs_info.targets[target_id])
        len_target = len(target)
        sliced_eventalign_df = reverse_eventalign(sliced_eventalign_df, len_target)
    
    if use_pc_majority:
        # print("Using pos+kmers majority for PC events...") 
        ### This reduces the variance around the line
        ### Useful for the XNA_4Ds samples, because not all of them have the same PC base
        ### What is effectively doing is removing the NNNNNN kmers, which have odd values
        
        pos_kmer_idx = (sliced_eventalign_df.groupby(['position','model_kmer'])
                        .size().sort_values().groupby(level=0).tail(1).index)
        focus_pc_idx = sliced_eventalign_df.set_index(['position','model_kmer']).index
        sliced_eventalign_df = sliced_eventalign_df[focus_pc_idx.isin(pos_kmer_idx)]
    
    return sliced_eventalign_df

def read_and_slice_eventalign(target_id, strand, eventalign_dir, refs_info,
                              use_pc_majority=False, margin=0, keep_only_ub=False,
                              file_tpl='{}_{}_eventalign.dat.gz'):
    from misc.data_io import read_eventalign
    
    # if target_id in refs_info1.targets_id:
    #     refs_info = refs_info1
    #     eventalign_dir = eventalign_dir1
    # elif target_id in refs_info2.targets_id:
    #     refs_info = refs_info2
    #     eventalign_dir = eventalign_dir2
    # else:
    #     raise ValueError()
    
    #### Reading data
    eventalign_file = file_tpl.format(target_id, strand)
    eventalign_filepath = os.path.join(eventalign_dir, eventalign_file)
    eventalign_df = read_eventalign(eventalign_filepath, verbose=False)
    eventalign_df.strand = strand
    
    if target_id == 'PC15':
        # raise NotImplementedError('PC15 target not tested yet')
        eventalign_df.loc[eventalign_df.position>24,'position'] += 1
        pos_24 = eventalign_df.loc[eventalign_df.position==24].copy()
        pos_24.position += 1
        eventalign_df = eventalign_df.append(pos_24).sort_values('position')
    
    sliced_eventalign_df = slice_eventalign(eventalign_df, refs_info, target_id, 
                                            strand='F', 
                                            kmer_len=6, 
                                            use_pc_majority=use_pc_majority, 
                                            margin=margin)
    
    sliced_eventalign_df = sliced_eventalign_df.reset_index(drop=True)
    
    if not target_id.startswith('PC') and keep_only_ub:
        ### Fixing reference kmers based on position (when ref does not contains N/X)
        ub_kmers, ub_kmer_pos = refs_info.get_ub_kmers(target_id, flat=True, return_pos=True)
        pos_kmer_map = dict(zip(ub_kmer_pos, ub_kmers))
        sliced_eventalign_df.reference_kmer = sliced_eventalign_df.apply(
            lambda x: pos_kmer_map.get(x.position, x.reference_kmer), axis=1)
        
        sliced_eventalign_df = sliced_eventalign_df[
            sliced_eventalign_df.reference_kmer.str.contains('N')]
    
    # if target_id == 'PC15':
    #     sliced_eventalign_df.loc[sliced_eventalign_df.position>24,'position'] += 1
    
    return sliced_eventalign_df.reset_index(drop=True)

def reverse_eventalign(eventalign_df, len_target, kmer_len=6):
    """
    Revert positions from eventalign df because it came from the reverse strand.
    Originally it contains the events not at the fast5 signal order.
    Instead it follows the position of the Forward strand.
    This can be verified by matching position with event_index:
        read_eventalign_df[['position','event_index']]

    Parameters
    ----------
    eventalign_df : pd.DataFrame
        DESCRIPTION.
    len_target : int
        DESCRIPTION.
    kmer_len : int, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    rev_eventalign_df : pd.DataFrame
        DESCRIPTION.

    """
    rev_eventalign_df = eventalign_df.copy()
    
    # rev_eventalign_df['position'] = eventalign_df.position*-1 + len_target - 1
    rev_eventalign_df['position'] = eventalign_df.position*-1 + len_target - kmer_len
    rev_eventalign_df.sort_values(['read_id','position','event_index'], inplace=True)
    ### \/ Seems most correct, but afraid of breaking something
    # rev_eventalign_df.sort_values(['read_id','event_index'], inplace=True) 
    rev_eventalign_df.reset_index(drop=True, inplace=True)
    
    return rev_eventalign_df

# def rereverse_eventalign(eventalign_df, len_target, kmer_len=6):
def unreverse_eventalign(eventalign_df, len_target, kmer_len=6):
    """
    Undo reverse_eventalign() function

    Parameters
    ----------
    eventalign_df : pd.DataFrame
        DESCRIPTION.
    len_target : int
        DESCRIPTION.
    kmer_len : int, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    rerev_eventalign_df : pd.DataFrame
        DESCRIPTION.

    """
    rev_eventalign_df = eventalign_df.copy()
    
    # rev_eventalign_df['position'] = eventalign_df.position*-1 + len_target - 1
    rev_eventalign_df['position'] = eventalign_df.position*-1 + len_target - kmer_len
    # rev_eventalign_df.sort_values(['read_id','event_index'],
    #                               ascending=[True, False], inplace=True)
    rev_eventalign_df.sort_values(['read_id','position','event_index'], 
                                  ascending=[True,True,False], inplace=True)
    rev_eventalign_df.reset_index(drop=True, inplace=True)
    
    return rev_eventalign_df

def unreverse_eventalign2(eventalign_df, len_target, kmer_len=6):
    rev_eventalign_df = eventalign_df.copy()
    
    ### Wrong because of + kmer_len
    rev_eventalign_df['position'] = eventalign_df.position*-1 + len_target + kmer_len
    ### ? Also wrong because of sorting by position?
    ### Need to sort by position because there are NaN vals at event_index after polishing
    rev_eventalign_df.sort_values(['read_id','position','event_index'], 
                                  ascending=[True,True,False], inplace=True)
    rev_eventalign_df.reset_index(drop=True, inplace=True)
    
    return rev_eventalign_df

def invert_samples(eventalign_df):
    """
    Invert order of signal values in the column samples.
    Used when dealing with eventalign from Reverse strand.

    Parameters
    ----------
    eventalign_df : pd.DataFrame

    Returns
    -------
    interted_eventalign_df : pd.DataFrame

    """
    inv_eventalign_df = eventalign_df.copy()
    
    inv_func = lambda x: ','.join( x.split(',')[::-1] )
    inv_eventalign_df['samples'] = inv_eventalign_df.samples.map(inv_func)
    
    return inv_eventalign_df

def extract_samples(eventalign_df):
    samples = np.asarray(','.join(eventalign_df.samples.values).split(','), 
                         dtype=float)
    return samples

def count_samples(eventalign_df, sum_all=False):
    num_samples = eventalign_df.samples.str.count(',') + 1
    if sum_all:
        num_samples = np.sum(num_samples)
    return num_samples

def extract_seq_samples(read_eventalign_df, x_pos, kmer_len=6, margin=3):
    seq_start = x_pos - kmer_len + 1 - margin
    seq_end = x_pos + margin
    
    seq_eventalign_df = read_eventalign_df[read_eventalign_df.position.between(
                                                           seq_start, seq_end)]
    
    target_id = read_eventalign_df.target_id.iloc[0]
    is_pc = target_id.startswith('PC')
    
    target_lst = []
    pos_lst = []
    is_pc_lst = []
    samples_lst = []
    for position, grp_df in seq_eventalign_df.groupby('position'):
        samples = extract_samples(grp_df).tolist()
        
        target_lst += [grp_df.target_id.iloc[0]] * len(samples)
        pos_lst += [position] * len(samples)
        is_pc_lst += [is_pc] * len(samples)
        samples_lst += samples
    
    seq_samples_df = pd.DataFrame({'target_id': target_lst,
                                   'position': pos_lst,
                                   'signal_level': samples_lst,
                                    'is_pc': is_pc_lst,
                                   # 'num_ubs': num_ubs_lst,
                                   })
    
    return seq_samples_df

def filter_demux(demux_df, read_len_interval=None, max_barcode_dist=None,
                 min_target_cover=None, use_tpl_coverage=True, 
                 min_target_acc=None, max_ub_area_acc=None, verbose=False,
                 read_type=None, output_dir=None):
    filt_demux_df = demux_df.copy()
    output_filename = 'demux-k_15-w_5'
    
    if read_type is not None:
        output_filename += '-{}_only'.format(read_type)
    
        print("Filtering by read type length:", read_type)
        
        prvs_size = filt_demux_df.shape[0]
        filt_demux_df = filt_demux_df[filt_demux_df.type==read_type.upper()]
        print("Removed {:>10,d}".format(prvs_size - 
                                        filt_demux_df.shape[0]))
        
        
    if read_len_interval is not None:
        min_length, max_length = read_len_interval
        output_filename += '-l_{}_{}'.format(min_length, max_length)
    
        print("Filtering by length:", min_length, "<= read_length <=", max_length)
        
        prvs_size = filt_demux_df.shape[0]
        # filt_demux_df = filt_demux_df[(filt_demux_df.read_length >= min_length) & 
        #                        (filt_demux_df.read_length <= max_length)]
        filt_demux_df = filt_demux_df[
            filt_demux_df.read_length.between(min_length, max_length)]
        print("Removed {:>10,d}".format(prvs_size - 
                                        filt_demux_df.shape[0]))
    
    if min_target_cover is not None:
        output_filename += '-t_{}'.format(min_target_cover)
        output_filename += '_tpl' if use_tpl_coverage else ''
        tc_key = 'template_coverage' if use_tpl_coverage else 'target_cover'
        print("Filtering by:", tc_key, ">=", min_target_cover)
        prvs_size = filt_demux_df.shape[0]
        filt_demux_df = filt_demux_df[filt_demux_df[tc_key] >= min_target_cover]
        print("Removed {:>10,d}".format(prvs_size - 
                                        filt_demux_df.shape[0]))
    
    # post_remove_dups = False
    # if not post_remove_dups:
    #     print("Duplicated read ids")
    #     prvs_size = filt_demux_df.shape[0]
    #     filt_demux_df.drop_duplicates('read_id', keep=False, inplace=True)
    #     print("Removed {:>10,d}".format(prvs_size - filt_demux_df.shape[0]))
    # else:
    #     print("Skipping removing duplicated read ids...")
    
    if max_barcode_dist is not None:
        output_filename += '-d_{}'.format(max_barcode_dist)
        print("Filtering by: barcode_distance <=", max_barcode_dist)
        prvs_size = filt_demux_df.shape[0]
        filt_demux_df = filt_demux_df[filt_demux_df.barcode_distance <= max_barcode_dist]
        print("Removed {:>10,d}".format(prvs_size - filt_demux_df.shape[0]))
    
    if min_target_acc is not None:
        output_filename += '-tar_acc_{}'.format(min_target_acc)
        print("Filtering by: target_acc >=", min_target_acc)
        prvs_size = filt_demux_df.shape[0]
        filt_demux_df = filt_demux_df[filt_demux_df['target_acc'] >= min_target_acc]
        print("Removed {:>10,d}".format(prvs_size - filt_demux_df.shape[0]))
    
    if max_ub_area_acc is not None:
        output_filename += '-ub_area_acc_{}'.format(max_ub_area_acc)
        print("Filtering by: ub_area_acc <=", max_ub_area_acc)
        prvs_size = filt_demux_df.shape[0]
        filt_demux_df = filt_demux_df[filt_demux_df['ub_area_acc'] <= max_ub_area_acc]
        print("Removed {:>10,d}".format(prvs_size - filt_demux_df.shape[0]))
    
    #### [opt] remove duplicated after demuxing. *Issue*: need to update merge on 'read_id'
    # if post_remove_dups:
    #     print("Removing demuxed duplicated read ids")
    #     prvs_size = filt_demux_df.shape[0]
    #     filt_demux_df.drop_duplicates('read_id', keep=False, inplace=True)
    #     print("Removed {:>10,d}".format(prvs_size - filt_demux_df.shape[0]))
    
    print("Size before and after filter:")
    print("{:>10,d}".format(demux_df.shape[0]))
    print("{:>10,d}".format(filt_demux_df.shape[0]))
    
    if output_dir is not None:
        output_filename += '.csv.gz'
        output_filepath = os.path.join(output_dir, output_filename)
        print("Saving:", output_filepath)
        filt_demux_df.to_csv(output_filepath)
    
    return filt_demux_df

def get_non_outliers(values, mode='sd', mode_arg=None):
    values = np.asarray(values)
    
    if mode == 'sd':
        mean = values.mean()
        std = values.std()
        std_mult = mode_arg if mode_arg is not None else 2
        lower_limit = mean - std_mult*std
        upper_limit = mean + std_mult*std
    elif mode == 'perc':
        perc_cut = mode_arg if mode_arg is not None else 0.02
        lower_limit = np.quantile(values, perc_cut)
        upper_limit = np.quantile(values, 1-perc_cut)
    else:
        raise ValueError(f"Invalid value for {mode=}, select [sd, perc]")
    
    return (values >= lower_limit) & (values <= upper_limit)

def normalize_med_mad(signal, factor=1.4826):
    ### Factor value retrieved from Bonito med_mad normalization
    med = np.median(signal)
    mad = np.median(np.absolute(signal - med)) * factor + np.finfo(np.float32).eps
    norm_signal = (signal - med) / mad
    return norm_signal

def normalize_med_mad_squiggly(clean_signal, stds, norm_rep=100, rng=np.random):
    rep_stds = np.repeat(stds, norm_rep)
    event_std = rng.uniform(-1*rep_stds, rep_stds)
    squiggly_signal = np.repeat(clean_signal, norm_rep) + event_std
    med = np.median(squiggly_signal)
    mad = np.median(np.absolute(squiggly_signal - med)) * 1.4826 + np.finfo(np.float32).eps
    norm_signal = (clean_signal - med) / mad
    return norm_signal

def convert_target_to_string(target, base_map=['N','A','C','G','T','X','Y']):
    return ''.join([ base_map[i] for i in target ])

def convert_string_to_target(target_str, base_map=['N','A','C','G','T','X','Y']):
    base_map_rev = dict(zip(base_map,range(len(base_map))))
    return [ base_map_rev[c] for c in target_str ]
