
import os
import gzip

import h5py
import pandas as pd
import numpy as np
# from ont_fast5_api.fast5_interface import get_fast5_file

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

DEFAULT_PAF_COLUMNS= ['read_id','read_length','read_start','read_end','strand',
                      'target_id','target_length','target_start','target_end',
                      'n_matches','block_length','mapping_quality']

def read_fasta(reads_filepath):
    if not reads_filepath.endswith('.gz'):
        with open(reads_filepath, 'r') as f:
            content = f.readlines()
    else:
        with gzip.open(reads_filepath, 'rt') as f: # Text mode (uses more mem)
            content = f.readlines()
    
    # print("WARNING: considering read_id with fixed length of 35.")
    reads_ids = [ info_line[1:37] for info_line in content[::2]]
    # print("reads_ids:", reads_ids)
    
    seqs = [ seq.strip('>\n') for seq in content[1::2]]
    # print(seqs[-1])
    
    # Sanity check of matching 1 to 1 
    assert len(reads_ids) == len(seqs)
    
    reads_dict = dict(zip(reads_ids, seqs))
    
    return reads_dict

def save_reads(reads_dict, reads_filepath):
    _, output_format = os.path.splitext(reads_filepath)
    
    with open(reads_filepath, "w") as out_handle:
        for read_id, seq in reads_dict.items():
            record = SeqRecord(
                Seq(seq),
                id=read_id,
                description="",
            )
            
            if output_format == '.fasta':
                SeqIO.write(record, out_handle, "fasta-2line")
            elif output_format == '.fastq':
                SeqIO.write(record, out_handle, "fastq")

def read_fastq(fastq_filepath, reads_ids=None):
    reads = []
    
    if reads_ids is not None:
        raise NotImplementedError()
        ### Check read id by record, remove it from list, like another func here
    
    for record in SeqIO.parse(fastq_filepath, "fastq"):
        read_seq = str(record.seq)
        reads.append(read_seq)
    
    return reads

def read_multiple_pafs(pafs_filepaths, **read_kwargs):
    pafs_list = [read_paf(paf_filepath, **read_kwargs)
                 for paf_filepath in pafs_filepaths]
    
    pafs_df = pd.concat(pafs_list).reset_index(drop=True)
    
    return pafs_df

def read_paf(paf_filepath, paf_columns=DEFAULT_PAF_COLUMNS, verbose=False,
             extra_tags=[], targets_list_filepath=None):
    if verbose:
        print("Reading paf file:", paf_filepath)
    
    if len(extra_tags) == 0:
        names = paf_columns
    else:
        first_row = pd.read_csv(paf_filepath, sep='\t' , header=None, 
                                index_col=False, nrows=1).squeeze("rows")
        
        tags_cols = []
        for tag in first_row.iloc[len(paf_columns):]:
            tag_name = tag.split(':')[0]
            tags_cols.append(tag_name)
        
        names = paf_columns + tags_cols
    
    usecols = paf_columns + extra_tags
    
    paf_df = pd.read_csv(paf_filepath, sep='\t' , names=names, 
                         usecols=usecols, index_col=False)
    
    
    if targets_list_filepath is not None:
        if verbose:
            print(f"Filtering paf by input targets list... (shape={paf_df.shape[0]:0,d})")
        # targets_list = pd.read_csv(targets_list_filepath, header=None, squeeze=True)
        targets_list = read_tsv(targets_list_filepath).read_id
        paf_df = paf_df[paf_df.target_id.isin(targets_list)]
        if verbose:
            print(f"* paf number of reads with selected targets: {paf_df.shape[0]:0,d}")
    
    ### Parsing extra tags
    for tag_name in extra_tags:
        paf_df[tag_name] = paf_df[tag_name].str.split(':', n=2).str[-1]
        ### TODO add casting to others types, if needed
    
    n_matches = paf_df['n_matches'].astype(float)
    target_covers = n_matches / paf_df['target_length'].astype(float)
    percent_matches = n_matches / paf_df['block_length'].astype(float)
    paf_df.loc[:, 'target_cover'] = target_covers # Non-official metric Nok defined
    paf_df.loc[:, 'percent_match'] = percent_matches # Usually called Identiy?
    
    paf_df.loc[:, 'read_alignment_length'] = paf_df.read_end - paf_df.read_start
    read_covers = n_matches / paf_df['read_alignment_length'].astype(float)
    paf_df.loc[:, 'read_alignment_cover'] = read_covers
    
    ### Real template/target coverage metric
    paf_df.loc[:, 'template_coverage'] = (
        paf_df.read_alignment_length / paf_df.target_length).clip(upper=1)
    
    paf_df['is_pc'] = False
    paf_df.loc[paf_df.target_id.str.startswith('PC'),'is_pc'] = True
    paf_df['type'] = "XNA"
    paf_df.loc[paf_df.is_pc, 'type'] = "PC"
    
    if verbose:
        print("paf number of alignments: {:0,d}".format(paf_df.shape[0]))
        print("  number of unique reads: {:0,d}".format(paf_df.read_id.unique().shape[0]))
    
    return paf_df

def get_read_seq(read_id, reads_filepath, read_info=None, reads_dict=None):
    ### This import is inside here to avoid error due to circular import
    from misc.utils import reverse_complement
    
    if not isinstance(read_id, list): # Single read_id
        if reads_dict is not None:
            read_seq = str(reads_dict[read_id].seq)
        elif reads_filepath.endswith('.hdf5'):
            with h5py.File(reads_filepath, 'r') as f_hdf:
                if read_id in f_hdf:
                    read_seq = f_hdf[read_id].asstr()[()]
                else:
                    read_seq = None
                    print("Warning: read_id not present in file", read_id)
        elif reads_filepath.endswith('.fasta'):
            read_seq = ''
            
            for record in SeqIO.parse(reads_filepath, "fasta"):
                if record.id == read_id:
                    read_seq = str(record.seq)
                    break
        elif reads_filepath.endswith('.fasta.gz'):
            read_seq = ''
            
            with gzip.open(reads_filepath, 'rt') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    if record.id == read_id:
                        read_seq = str(record.seq)
                        break
        elif reads_filepath.endswith('.fastq'):
            read_seq = ''
            
            for record in SeqIO.parse(reads_filepath, "fastq"):
                if record.id == read_id:
                    read_seq = str(record.seq)
                    break
        else:
            raise ValueError("Unknown format for reads_filepath: "+reads_filepath)
    else: # It is a list, load multiple reads
        read_seq = {}
        
        if reads_filepath.endswith('.hdf5'):
            with h5py.File(reads_filepath, 'r') as f_hdf:
                for r_id in read_id:
                    if r_id in f_hdf:
                        r_seq = f_hdf[r_id].asstr()[()]
                        read_seq[r_id] = r_seq
                    else:
                        print("Warning: read_id not present in file", r_id)
        elif reads_filepath.endswith('.fasta'):
            for record in SeqIO.parse(reads_filepath, "fasta"):
                if record.id in read_id:
                    read_seq[record.id] = str(record.seq)
                    read_id.remove(record.id)
                if len(read_id) == 0:
                    break
        elif reads_filepath.endswith('.fasta.gz'):
            with gzip.open(reads_filepath, 'rt') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    if record.id in read_id:
                        read_seq[record.id] = str(record.seq)
                        read_id.remove(record.id)
                    if len(read_id) == 0:
                        break
        elif reads_filepath.endswith('.fastq'):
            for record in SeqIO.parse(reads_filepath, "fastq"):
                if record.id in read_id:
                    read_seq[record.id] = str(record.seq)
                    read_id.remove(record.id)
                if len(read_id) == 0:
                    break
        else:
            raise ValueError("Unknown format for reads_filepath: "+reads_filepath)
    
    
    if read_info is not None:
        assert len(read_seq) == read_info.read_length # Sanity check to validade that retrieved read matches alignment read_info
        
        read_seq = read_seq[read_info.read_start:read_info.read_end]

        if read_info.strand in ['-','R']:
            read_seq = reverse_complement(read_seq)
    
    return read_seq

def get_read_qual(read_id, reads_filepath, read_info=None, reads_dict=None):
    if not isinstance(read_id, list): # Single read_id
        if reads_dict is not None:
            # read_seq = str(reads_dict[read_id].seq)
            
            if read_id not in reads_dict:
                raise KeyError(f"Read not found: {read_id}")
            
            record = reads_dict[read_id]
        elif reads_filepath.endswith('.fastq'):
            record = None
            
            for record in SeqIO.parse(reads_filepath, "fastq"):
                if record.id == read_id:
                    # read_seq = str(record.seq)
                    break
            
            if record is None:
                raise KeyError(f"Read not found: {read_id}")
        else:
            raise ValueError("Unknown format for reads_filepath: "+reads_filepath)
        read_qual = record.letter_annotations["phred_quality"]
    else: # It is a list, load multiple reads
        read_qual = {}
        
        if reads_filepath.endswith('.fastq'):
            for record in SeqIO.parse(reads_filepath, "fastq"):
                if record.id in read_id:
                    read_qual[record.id] = record.letter_annotations["phred_quality"]
                    read_id.remove(record.id)
                if len(read_id) == 0:
                    break
        else:
            raise ValueError("Unknown format for reads_filepath: "+reads_filepath)
    
    
    if read_info is not None:
        assert len(read_qual) == read_info.read_length # Sanity check to validade that retrieved read matches alignment read_info
        
        read_qual = read_qual[read_info.read_start:read_info.read_end]

        if read_info.strand in ['-','R']:
            read_qual = read_qual[::-1]
    
    return read_qual

def read_ref_fasta(reference_filepath, selected_ids=None):
    # print("Reading refdb file...")
    # print(reference_filepath)
    
    if selected_ids is None:
        if not reference_filepath.endswith('.gz'):
            with open(reference_filepath, 'r') as f:
                content = f.readlines()
        else:
            with gzip.open(reference_filepath, 'rt') as f: # Text mode (uses more mem)
                content = f.readlines()
        
        # with open(reference_filepath, 'r') as ref_f:
        #     content = ref_f.readlines()
        
        target_ids = [ t_id.strip('>\n') for t_id in content[::2]]
        # print("target_ids:", target_ids)
        
        targets = [ tar.strip('>\n') for tar in content[1::2]]
        # print(targets[-1])
        
        # Sanity check of matching 1 to 1 
        assert len(target_ids) == len(targets)
        
        refdb = dict(zip(target_ids, targets))
    else:
        if not reference_filepath.endswith('.gz'):
            open_func = open
            open_mode = 'r'
        else:
            open_func = gzip.open
            open_mode = 'rt'
        
        refdb = {}
        with open_func(reference_filepath, open_mode) as f_in:
            target_id_line = f_in.readline()
            
            # count = 0
            while target_id_line != '':
            # while target_id_line != '' and count < 10:
                target_id = target_id_line.strip('>\n')
                target = f_in.readline().strip('>\n')
                
                # print(target_id)
                if target_id in selected_ids:
                    refdb[target_id] = target
                
                target_id_line = f_in.readline()
                # count += 1
    
    return refdb

def read_demux(demux_filepath, sample_list_filepath=None, 
               exclude_list_filepath=None, 
               include_list_filepath=None, 
               verbose=False):

    if verbose:
        print("Reading demux file:", demux_filepath)
        if sample_list_filepath is not None: print(f"{sample_list_filepath=}")
        if exclude_list_filepath is not None: print(f"{exclude_list_filepath=}")
        if include_list_filepath is not None: print(f"{include_list_filepath=}")
    
    demux_df = pd.read_csv(demux_filepath, index_col=0)
    
    if verbose:
        print("demux number of reads: {:0,d}".format(demux_df.shape[0]))
    
    if exclude_list_filepath is not None:
        ids_list = read_tsv(exclude_list_filepath).read_id
        count_matches = demux_df.index.isin(ids_list).sum()
        demux_df = demux_df[~demux_df.index.isin(ids_list)]
        if verbose:
            print("Filtering reads present in exclude list:", 
                  count_matches, 
                  "out of", len(ids_list), "on list.")
            print("remaining number of reads: {:0,d}".format(demux_df.shape[0]))
            
    if include_list_filepath is not None:
        ids_list = read_tsv(include_list_filepath).read_id
        count_matches = demux_df.index.isin(ids_list).sum()
        demux_df = demux_df[demux_df.index.isin(ids_list)]
        # demux_df = demux_df.loc[ids_list]
        if verbose:
            print("Keeping reads present in include list:",
                  f"{count_matches:0,d} out of {len(ids_list):0,d} on list.")
            # print("Keeping reads present in include list:", 
            #       demux_df.index.isin(ids_list).sum(), 
            #       "out of", len(ids_list), "on list.")
        
    if sample_list_filepath is not None:
        # sample_list = pd.read_csv(sample_list_filepath).read_id
        sample_list = read_tsv(sample_list_filepath).read_id
        # demux_df = pd.read_csv(demux_filepath, index_col=0,
        #                        skiprows=lambda idx: idx not in sample_list)
        demux_df = demux_df.loc[sample_list]
        
        if verbose:
            print("Sampling demux...")
            print("total number of reads: {:0,d}".format(demux_df.shape[0]))
    
    if 'barcode_name' in demux_df:
        demux_df.rename(columns={"barcode_name": "target_id"}, inplace=True)
    
    if 'is_pc' not in demux_df:
        demux_df['is_pc'] = False
        demux_df.loc[demux_df.target_id.str.startswith('PC'),'is_pc'] = True
    if 'type' not in demux_df:
        demux_df['type'] = "XNA"
        demux_df.loc[demux_df.is_pc, 'type'] = "PC"
    
    if 'read_alignment_cover' not in demux_df:
        demux_df.loc[:, 'read_alignment_length'] = demux_df.read_end - demux_df.read_start
        n_matches = demux_df['n_matches'].astype(float)
        read_covers = n_matches / demux_df['read_alignment_length'].astype(float)
        demux_df.loc[:, 'read_alignment_cover'] = read_covers
        
    if 'template_coverage' not in demux_df:
        demux_df.loc[:, 'template_coverage'] = (
            demux_df.read_alignment_length / demux_df.target_length).clip(upper=1)
    
    
    return demux_df

def read_eventalign(eventalign_filepath, verbose=False, reverse=False, 
                    target_len=None, sample_list_filepath=None,
                    target_id_strand=None, file_tpl='{}_{}_eventalign.dat.gz',
                    check_and_fix_reverse=True):
    ### This import is inside here to avoid error due to circular import
    from misc.utils import reverse_complement
    
    if target_id_strand is not None:
        target_id, strand= target_id_strand
        eventalign_filepath = os.path.join(eventalign_filepath,
                                           file_tpl.format(target_id, strand))
    
    eventalign_df = pd.read_csv(eventalign_filepath, sep='\t')
    # eventalign_df = pd.read_csv(eventalign_filepath, sep='\t', index_col=0)
    
    # eventalign_df.drop(columns='Unnamed: 0', inplace=True, errors='ignore')
    eventalign_df.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True, errors='ignore')
    
    eventalign_df.rename(columns={"contig": "target_id",
                                  "read_name": "read_id",}, 
                         inplace=True)
    
    if sample_list_filepath is not None:
        ids_list = read_tsv(sample_list_filepath).read_id
        
        if verbose:
            print("Sampling eventalign using list...")
            # print("Filtering reads present in sample list:", 
            #       eventalign_df.read_id.isin(ids_list).sum(), 
            #       "out of", len(ids_list), "on list.")
        eventalign_df = eventalign_df[eventalign_df.read_id.isin(ids_list)]
        if verbose:
            print("Filtering reads present in sample list:", 
                  eventalign_df.read_id.nunique(), 
                  "out of", len(ids_list), "on list.")
    
    if reverse:
        eventalign_df['position'] = eventalign_df.position*-1 + target_len - 1
        ### Any info about target_len in eventalign df itself?
    
    if check_and_fix_reverse and eventalign_df.event_index.isna().any():
        if verbose:
            print("Checking if reference_kmer needs fixing(rvl-cpl)...")
        ### This check assumes df is sorted by position
        need_fix = False
        # for row_id, row in eventalign_df.iterrows():
        for row_id, row in eventalign_df[eventalign_df.event_index.isna()].iterrows():
            # if 'N' in row.reference_kmer:
            #     next_row = eventalign_df.loc[row_id+1]
            #     # print(row.position, row.reference_kmer)
            #     ### Need fix if first occurrence starts with N 
            #     ### Sanity checks, making sure it is sorted
            #     ### (prone to fail at some cases: missing position, ?)
            #     if row.reference_kmer.startswith('N'):
            #         need_fix = True
            #         assert row.reference_kmer[:-1] == next_row.reference_kmer[1:]
            #     else:
            #         assert row.reference_kmer.endswith('N')
            #         assert row.reference_kmer[1:] == next_row.reference_kmer[:-1]
                
            #     break
            
            # if 'N' in row.reference_kmer and np.isnan(row.event_index):
            if 'N' in row.reference_kmer:
                ### Only checking ub kmers (safer) AND that were modified
                next_row = eventalign_df.loc[row_id+1]
                # print(row.position, row.reference_kmer)
                
                if row.position == next_row.position-1: # Check if next position
                    if row.reference_kmer[:-1] == next_row.reference_kmer[1:]:
                        ### Shifting on wrong direction (65432_1 -> 7_65432)
                        need_fix = True
                    else:
                        ### Sanity checks, making sure it is sorted
                        ### Shifting on right direction (1_23456 -> 23456_7)
                        assert row.reference_kmer[1:] == next_row.reference_kmer[:-1]
                
                    break
            
        if need_fix:
            if verbose:
                print("[WARNING] Fixing reversed reference_kmers...")
            to_fix_mask = eventalign_df.event_index.isna()
            eventalign_df.loc[to_fix_mask, 'reference_kmer'] = \
                eventalign_df[to_fix_mask].reference_kmer.apply(reverse_complement)
    
    if verbose:
        # print("> eventalign size: {:0,d}".format(eventalign_df.shape[0]))
        # print("> unique read ids: {:0,d}".format(eventalign_df.read_id.unique().shape[0]))
        print("> unique read ids: {:0,d}".format(eventalign_df.read_id.nunique()))
    
    return eventalign_df

def get_fast5_reads_ids(fast5_filepath):
    from ont_fast5_api.fast5_interface import get_fast5_file
    
    with get_fast5_file(fast5_filepath, mode='r') as f5_file:
        read_ids = f5_file.get_read_ids()
    
    return read_ids

def get_raw_data(read_id, fast5_filepath):
    from ont_fast5_api.fast5_interface import get_fast5_file
    
    with get_fast5_file(fast5_filepath, mode='r') as f5_file:
        fast5Read = f5_file.get_read(read_id)
        raw_data = fast5Read.get_raw_data(scale=True)
    
    return raw_data

def read_sam(sam_filepath, verbose=False):
    import pysam
    
    if sam_filepath.endswith('.sam'):
        samfile = pysam.AlignmentFile(sam_filepath, "r")
    elif sam_filepath.endswith('.bam'):
        samfile = pysam.AlignmentFile(sam_filepath, "rb")
   
    column_names = [
        'read_id', 'read_length', 'read_start', 'read_end', 'strand',
        'target_id', 'target_length', 'target_start', 'target_end', 'n_matches',
        'read_alignment_length', 'mapping_quality']
    data = dict(zip(column_names, [ [] for i in range(len(column_names))]))
    for read in samfile:
        data['read_id'].append(read.query_name)
        data['read_length'].append(read.query_length)
        data['read_start'].append(read.query_alignment_start)
        data['read_end'].append(read.query_alignment_end)
        data['strand'].append('+' if not read.is_reverse else '-')
        data['target_id'].append(read.reference_name)
        data['target_length'].append(samfile.lengths[read.reference_id])
        data['target_start'].append(read.reference_start)
        data['target_end'].append(read.reference_end)
        data['n_matches'].append(np.sum([ t[1] for t in read.cigartuples if t[0] == 0 ]))
        data['read_alignment_length'].append(read.query_alignment_length)
        data['mapping_quality'].append(read.mapping_quality)
        # data['xxxxx'].append(xxxxx)
        # read.reference_length = aligned length of the read on the reference genome
    samfile.close()
        
    sam_df = pd.DataFrame(data=data)
   
    n_matches = sam_df['n_matches'].astype(float)
    target_covers = n_matches / sam_df['target_length'].astype(float)
    sam_df.loc[:, 'target_cover'] = target_covers
    
    #### TODO Find a way to compute block_length and/or percent_match
    # percent_matches = n_matches / sam_df['block_length'].astype(float)
    # sam_df.loc[:, 'percent_match'] = percent_matches
    
    sam_df.loc[:, 'read_alignment_length'] = sam_df.read_end - sam_df.read_start
    read_covers = n_matches / sam_df['read_alignment_length'].astype(float)
    sam_df.loc[:, 'read_alignment_cover'] = read_covers
    
    sam_df['is_pc'] = False
    sam_df.loc[sam_df.target_id.str.startswith('PC'),'is_pc'] = True
    sam_df['type'] = "XNA"
    sam_df.loc[sam_df.is_pc, 'type'] = "PC"
    
    # sam_df['block_length'] = np.nan
    # sam_df['percent_match'] = np.nan
    sam_df['block_length'] = sam_df['read_alignment_length']
    sam_df['percent_match'] = sam_df['read_alignment_cover']
    
    if verbose:
        print("paf number of alignments: {:0,d}".format(sam_df.shape[0]))
        print("  number of unique reads: {:0,d}".format(sam_df.read_id.unique().shape[0]))
    
    return sam_df
    
def manual_read_sam(sam_filepath, verbose=False):
    sam_columns = [
        'read_id', 'FLAG', 'target_id', 'target_start',
        'mapping_quality', 'CIGAR', 
        'RNEXT', # This field is set as '*' when the information is unavailable
        'PNEXT', # Set as 0 when the information is unavailable
        'TLEN', # It is set as 0 for single-segment template or when the information is unavailable.
        'seq', 'QUAL']
    
    names = sam_columns
    usecols = sam_columns
    # usecols = sam_columns[:-1] # Not using QUAL
    
    skiprows = 0
    with open(sam_filepath, 'r') as sam_file:
        line = sam_file.readline()
        while line.startswith(('@','[')):
            skiprows += 1
            line = sam_file.readline()
    
    sam_df = pd.read_csv(sam_filepath, sep='\t' , names=names, skiprows=skiprows,
                         # engine='python',
                         usecols=usecols,
                         index_col=False)
    
    
    is_reversed = (sam_df.FLAG & 16) == 16
    
    sam_df['strand'] = '+'
    sam_df.loc[is_reversed, 'strand'] = '-'
    
    # sam_df['target_length'] = '+'
    
    return sam_df

def read_tsv(tsv_filepath):
    # first_row = pd.read_csv(tsv_filepath, index_col=0, nrows=0).columns.tolist()
    first_row = pd.read_csv(tsv_filepath, index_col=None, nrows=0).columns.tolist()
    if 'read_id' in first_row:
        tsv_df = pd.read_csv(tsv_filepath)
    else:
        # tsv_df = pd.read_csv(tsv_filepath, header=None)
        tsv_df = pd.read_csv(tsv_filepath, names=['read_id'])
    
    return tsv_df

def index_reads_file(reads_filepath, verbose=False):
    if verbose:
        print("Indexing reads file... ", end='')
    
    reads_dict = SeqIO.index(reads_filepath, reads_filepath.split('.')[-1])
    
    if verbose:
        print(f"Number of reads: {len(reads_dict):,d}")
    
    return reads_dict


def read_ctc_data(ctc_dir, bkps=False):
    chunks = np.load(os.path.join(ctc_dir, 'chunks.npy'), mmap_mode='r')
    targets = np.load(os.path.join(ctc_dir, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(ctc_dir, "reference_lengths.npy"), mmap_mode='r')
    
    indices = os.path.join(ctc_dir, "indices.npy")
    if os.path.exists(indices):
        # print("[indices.npy found: using idx for subsampling]")
        print("[WARNING] indices.npy found, using idx for selecting chunks")
        idx = np.load(indices)
        chunks = chunks[idx,:]
        targets = targets[idx,:]
        lengths = lengths[idx]
    
    if bkps:
        bkps = np.load(os.path.join(ctc_dir, "breakpoints.npy"), mmap_mode='r')
        if os.path.exists(indices):
            bkps = bkps[idx,:]
        
        return chunks, targets, lengths, bkps
    return chunks, targets, lengths

def save_ctc_data(chunks, lengths, targets, output_directory, shuffle=True, bkps=None,
                  chunks_dtype=np.float16):
    chunks = np.asarray(chunks, dtype=chunks_dtype) # np.float32
    # chunks = np.asarray(chunks, dtype=chunks.dtype)
    # print(f"{chunks.dtype=}")
    
    if len(np.shape(targets)) == 1: # List of variable length arrays, need to pad before stacking
        targets_ = np.zeros((chunks.shape[0], max(lengths)), dtype=np.uint8)
        for idx, target in enumerate(targets): targets_[idx, :len(target)] = target
    else: # numpy matrix
        targets_ = targets
    lengths = np.asarray(lengths, dtype=np.uint16)
    
    if shuffle:
        indices = np.random.permutation(np.arange(len(lengths)))
    
        chunks = chunks[indices]
        targets_ = targets_[indices]
        lengths = lengths[indices]
        
        if bkps is not None:
            bkps = bkps[indices]
    
    # summary = pd.read_csv(summary_file(), sep='\t')
    # summary.iloc[indices].to_csv(summary_file(), sep='\t', index=False)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Saving ctc-data to:", output_directory)
    
    np.save(os.path.join(output_directory, "chunks.npy"), chunks)
    np.save(os.path.join(output_directory, "references.npy"), targets_)
    np.save(os.path.join(output_directory, "reference_lengths.npy"), lengths)
    if bkps is not None:
        np.save(os.path.join(output_directory, "breakpoints.npy"), bkps)
    
    print("Saved successfully!")
    
BASE_DIR = 'D:\\GIS\\' # Home Notebook
if not os.path.exists(BASE_DIR):
    BASE_DIR = 'C:\\Data\\' # GIS Notebook
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.expanduser('~/projects/xna_basecallers/') # Generic
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.expanduser('~/projects/xna_basecallers/') # Nok GPU
if not os.path.exists(BASE_DIR):
    BASE_DIR = '/home/ubuntu/projects/xna_basecallers/' # Ronin mauriciolp2
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError("No base dir found.")

# DEFAULT_MODEL = os.path.join(BASE_DIR, 'misc', 'r9.4_450bps.nucleotide.6mer.template.model')
def load_kmer_poremodel(ref_filepath=None, kmer_len=6):
    if ref_filepath is None:
        # ref_filepath = DEFAULT_MODEL
        ref_filepath = os.path.join(BASE_DIR, 'misc', 
                                    f'r9.4_450bps.nucleotide.{kmer_len}mer.template.model')
    model_kmer_df = pd.read_csv(ref_filepath, sep='\t', comment='#')
    tight_dict = model_kmer_df.set_index('kmer')[['level_mean','level_stdv']].to_dict('split')
    kmer_poremodel = { k:v for k,v in zip(tight_dict['index'], tight_dict['data']) }
    return kmer_poremodel