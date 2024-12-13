import os, re
from itertools import chain

from numpy.lib.stride_tricks import sliding_window_view

from misc.data_io import read_ref_fasta
from misc.utils import reverse_complement

REFS_DIR = './xna_libs'
if not os.path.exists(REFS_DIR):
    raise FileNotFoundError(f"References dir not found: {REFS_DIR}")

VALID_REFS = ['POC','CPLX','XNA16','XNA_4Ds','XNA1024','XNA1024-A027', 'XNA20']

EXP_REF_MAP = {
    'POC':'POC', 'CPLX':'CPLX', 
    'A003':'XNA16', 
    'A007':'XNA_4Ds', 'A008':'XNA_4Ds', 'A007+A008':'XNA_4Ds', 
    'A026':'XNA1024', 'A027':'XNA1024-A027', 'A026+A027':'XNA1024', 
    'XNA20':'XNA20', 
}

VALID_EXPS = list(EXP_REF_MAP.keys())

REF_EXP_MAP = { k:[] for k in set(EXP_REF_MAP.values())}
for k,v in EXP_REF_MAP.items(): REF_EXP_MAP[v].append(k)

class XNA_refs(object):
    def __init__(self, ref_name, short_version=True, ref_file=None, use_aliases=False,
                 refs_dir=REFS_DIR):
        # ref_name = ref_name.upper()
        
        if ref_name not in VALID_REFS:
            raise ValueError("Invalid ref_name ({}), choose among: {}".format(
                ref_name, VALID_REFS))
        
        self.ref_name = ref_name
        self.left_arm_len = 1214
        self.right_arm_len = 1386
        self.left_arm = ('ATCATTGCAGCACTGGGGCCAGATGGTAAGCCCTCCCGTATCGTAGTTATCT'
             'ACACGACGGGGAGTCAGGCAACTATGGATGAACGAAATAGACAGATCGCTGAGATAGGTGCCTC'
             'ACTGATTAAGCATTGGTAACTGTCAGACCAAGTTTACTCATATATACTTTAGATTGATTTAAAA'
             'CTTCATTTTTAATTTAAAAGGATCTAGGTGAAGATCCTTTTTGATAATCTCATGACCAAAATCC'
             'CTTAACGTGAGTTTTCGTTCCACTGAGCGTCAGACCCCGTAGAAAAGATCAAAGGATCTTCTTG'
             'AGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTG'
             'GTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGC'
             'AGATACCAAATACTGTTCTTCTAGTGTAGCCGTAGTTAGGCCACCACTTCAAGAACTCTGTAGC'
             'ACCGCCTACATACCTCGCTCTGCTAATCCTGTTACCAGTGGCTGCTGCCAGTGGCGATAAGTCG'
             'TGTCTTACCGGGTTGGACTCAAGACGATAGTTACCGGATAAGGCGCAGCGGTCGGGCTGAACGG'
             'GGGGTTCGTGCACACAGCCCAGCTTGGAGCGAACGACCTACACCGAACTGAGATACCTACAGCG'
             'TGAGCTATGAGAAAGCGCCACGCTTCCCGAAGGGAGAAAGGCGGACAGGTATCCGGTAAGCGGC'
             'AGGGTCGGAACAGGAGAGCGCACGAGGGAGCTTCCAGGGGGAAACGCCTGGTATCTTTATAGTC'
             'CTGTCGGGTTTCGCCACCTCTGACTTGAGCGTCGATTTTTGTGATGCTCGTCAGGGGGGCGGAG'
             'CCTATGGAAAAACGCCAGCAACGCGGCCTTTTTACGGTTCCTGGCCTTTTGCTGGCCTTTTGCT'
             'CACATGTTCTTTCCTGCGTTATCCCCTGATTCTGTGGATAACCGTATTACCGCCTTTGAGTGAG'
             'CTGATACCGCTCGCCGCAGCCGAACGACCGAGCGCAGCGAGTCAGTGAGCGAGGAAGCGGAAGA'
             'GCGCCCAATACGCAAACCGCCTCTCCCCGCGCGTTGGCCGATTCATTAATGCAGCTGGCACGAC'
             'AGGTTTCCCGACTGGAAAGCGGGCAGTGAGCGCAACGCAATTAATGTGAGTTAGCTCACTCATT'
             'AGGCACCCCA')
        self.right_arm = ('CTATGACCATGATTACGCCAAGCTTGCATGCCTGCAGGTCGACTCTAGAGG'
              'ATCCCCGGGTACCGAGCTCGAATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACC'
              'CTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCG'
              'AAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCGCCTGA'
              'TGCGGTATTTTCTCCTTACGCATCTGTGCGGTATTTCACACCGCATATGGTGCACTCTCAGTA'
              'CAATCTGCTCTGATGCCGCATAGTTAAGCCAGCCCCGACACCCGCCAACACCCGCTGACGCGC'
              'CCTGACGGGCTTGTCTGCTCCCGGCATCCGCTTACAGACAAGCTGTGACCGTCTCCGGGAGCT'
              'GCATGTGTCAGAGGTTTTCACCGTCATCACCGAAACGCGCGAGACGAAAGGGCCTCGTGATAC'
              'GCCTATTTTTATAGGTTAATGTCATGATAATAATGGTTTCTTAGACGTCAGGTGGCACTTTTC'
              'GGGGAAATGTGCGCGGAACCCCTATTTGTTTATTTTTCTAAATACATTCAAATATGTATCCGC'
              'TCATGAGACAATAACCCTGATAAATGCTTCAATAATATTGAAAAAGGAAGAGTATGAGTATTC'
              'AACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCCTTCCTGTTTTTGCTCACC'
              'CAGAAACGCTGGTGAAAGTAAAAGATGCTGAAGATCAGTTGGGTGCACGAGTGGGTTACATCG'
              'AACTGGATCTCAACAGCGGTAAGATCCTTGAGAGTTTTCGCCCCGAAGAACGTTTTCCAATGA'
              'TGAGCACTTTTAAAGTTCTGCTATGTGGCGCGGTATTATCCCGTATTGACGCCGGGCAAGAGC'
              'AACTCGGTCGCCGCATACACTATTCTCAGAATGACTTGGTTGAGTACTCACCAGTCACAGAAA'
              'AGCATCTTACGGATGGCATGACAGTAAGAGAATTATGCAGTGCTGCCATAACCATGAGTGATA'
              'ACACTGCGGCCAACTTACTTCTGACAACGATCGGAGGACCGAAGGAGCTAACCGCTTTTTTGC'
              'ACAACATGGGGGATCATGTAACTCGCCTTGATCGTTGGGAACCGGAGCTGAATGAAGCCATAC'
              'CAAACGACGAGCGTGACACCACGATGCCTGTAGCAATGGCAACAACGTTGCGCAAACTATTAA'
              'CTGGCGAACTACTTACTCTAGCTTCCCGGCAACAATTAATAGACTGGATGGAGGCGGATAAAG'
              'TTGCAGGACCACTTCTGCGCTCGGCCCTTCCGGCTGGCTGGTTTATTGCTGATAAATCTGGAG'
              'CCGGTGAGCGTG')
        
        self.short_version = short_version
        if ref_name == 'XNA16':
            self.barcode_len = 24
            self.left_primer_len = 25 # TTTTTTTTGCGTAGCGGGATCCAGC
            ## PC15 is an exception, left_primer_len = 24 (-C)
            ## PC16 has different seq  (TTTTTTTTGCGTAGCGGGATCCAGA)
            self.middle_primer_len = 24 # ACGATAATACGACTCACTATAGGG
            self.right_primer_len = 26 # CCGTCATAGCTGTTTCCTGTGTGAAA
            self.left_primer = 'TTTTTTTTGCGTAGCGGGATCCAGC'
            self.middle_primer = 'ACGATAATACGACTCACTATAGGG'
            self.right_primer = 'CCGTCATAGCTGTTTCCTGTGTGAAA'
        elif ref_name == 'XNA_4Ds': # XNA_4DS
            self.barcode_len = 24
            self.left_primer_len = 25 # TTTTTTTTGCGTAGCGGGATCCAGC
            self.middle_primer_len = 19 # ACGATAATACGACTCACTA
            self.right_primer_len = 23 # TCATAGCTGTTTCCTGTGTGAAA
        # elif ref_name == 'XNA1024' or ref_name == 'XNA1024-A027':
        elif ref_name.startswith('XNA1024') or ref_name == 'CPLX':
            self.barcode_len = 30
            self.left_primer_len = 23 # TTTTTTGCGTAGCGGTATGCGTA
            self.middle_primer_len = 2 # AT
            self.right_primer_len = 23 # TATGGCAGCTGTTTCATGTGTGA
            self.left_primer = 'TTTTTTGCGTAGCGGTATGCGTA'
            self.middle_primer = 'AT'
            self.right_primer = 'TATGGCAGCTGTTTCATGTGTGA'
            self.targets_id_16 = [
                'AAATC', 'AAATT', 'AATCC', 'AATCT', 'AATTC', 'AATTT', 'AGATC', 'AGATT', 
                'CGCAA', 'CTAAA', 'CTACA', 'CTACC', 'CTCAA', 'GAATA', 'GACAA', 'GATCA', 
                'GATCC', 'GATTA', 'GCCAA', 'GCGAA', 'GCGCA', 'GCGCC', 'GGATA', 'GGATC', 
                'GGATT', 'GGCAA', 'GGCAC', 'GGCGA', 'GGGAA', 'GGGCA', 'GGGCC', 'GGTAA', 
                'GTAAA', 'GTAAC', 'GTACA', 'GTACC', 'GTAGA', 'GTATC', 'GTATT', 'GTCAA', 
                'GTCAC', 'GTCGA', 'GTGAA', 'GTGCA', 'GTGCC', 'GTGGA', 'TAATC', 'TACAA', 
                'TATCC', 'TATTC', 'TGATC']
            self.targets_id_4ds = [
                'ACGTA', 'ACGTC', 'ACGTG', 'ACTGT', 'CGATA', 'CGATC', 'CGATG', 'CGTAT', 
                'GCATA', 'GCATC', 'GCATG', 'GCGAT', 'TCAGA', 'TCATG', 'TCGAC', 'TCGAT']
            self.targets_id_20 = self.targets_id_16 + self.targets_id_4ds
        elif ref_name in ['XNA20','POC'] :
            self.refs = dict(
                XNA16=XNA_refs('XNA16', short_version=short_version), 
                XNA_4Ds=XNA_refs('XNA_4Ds', short_version=short_version, use_aliases=True))
            self.barcode_len = 24
            self.left_primer_len = 25 # TTTTTTTTGCGTAGCGGGATCCAGC
            
            ref_16 = self.refs['XNA16']
            ref_4ds = self.refs['XNA_4Ds']
            
            attrs_to_merge = set(vars(ref_16).keys()).intersection(
                             set(vars(ref_4ds).keys()))
            
            for attr in attrs_to_merge:
                # print(attr)
                attr_16 = getattr(ref_16, attr)
                attr_4ds = getattr(ref_4ds, attr)
                
                if isinstance(attr_16, list):
                    attr_merged = attr_16 + attr_4ds
                elif isinstance(attr_16, dict):
                    attr_merged = attr_16.copy()
                    attr_merged.update(attr_4ds)
                else:
                    # print('> skipped')
                    continue
                    # raise NotImplementedError(f"{attr=} {type(attr_16)}")
                setattr(self, attr, attr_merged)
            return None
        
        if ref_file is None:
            if short_version:
                ref_file = 'refdb_short.fasta'
            else:
                ref_file = 'refdb.fasta'
        
        ref_filepath = os.path.join(refs_dir, ref_name.split('-')[0], ref_file)
        
        if ref_name == 'XNA1024-A027':
            ref_filepath = os.path.join(refs_dir,'XNA1024','refdb_short-v3-A027.fasta')
        
        self.ref_filepath = ref_filepath
        #### Reading .fasta
        self.targets = read_ref_fasta(ref_filepath)
        
        if ref_name == 'XNA_4Ds':
            self.aliases = dict(
                XNA17='84Ds4-AA', PC17='PC_84Ds4-AA',
                XNA18='84Ds4-AB', PC18='PC_84Ds4-AB',
                XNA19='84Ds4-AC', PC19='PC_84Ds4-AC',
                XNA20='84Ds4-AD', PC20='PC_84Ds4-AD',
            )
            self.aliases_targets_id = list(self.aliases.keys())
            
            self.aliases.update({v: k for k, v in self.aliases.items()})
            
            
            if not any([t.startswith('PC') for t in self.targets.keys()]):
                for tar_id in list(self.targets.keys()):
                    self.targets['PC_' + tar_id] = self.targets[tar_id]
            
            if use_aliases:
                self.targets = { self.aliases[k]:v for k,v in self.targets.items() }
        elif ref_name == 'XNA1024-16':
            self.targets = {k:v for k,v in self.targets.items()
                            if k in self.targets_id_16}
        elif ref_name == 'XNA1024-4Ds':
            self.targets = {k:v for k,v in self.targets.items()
                            if k in self.targets_id_4ds}
        elif ref_name == 'XNA1024-20':
            self.targets = {k:v for k,v in self.targets.items()
                            if k in self.targets_id_16 + self.targets_id_4ds}
                
        self.targets_id = list(self.targets.keys())
        self.xna_targets_id = [ t_id for t_id in self.targets_id if not t_id.startswith('PC') ]
        self.pc_targets_id = [ t_id for t_id in self.targets_id if t_id.startswith('PC') ]
        self.targets_len = list(set([ len(t) for t in self.targets.values()]))
        self.long_targets_len = [ self.left_arm_len + l + self.right_arm_len
                                 for l in self.targets_len ]
        
        ### add PC targets to 'XNA_4DS'
        # if ref_name == 'XNA_4Ds' and not any([t.startswith('PC') 
        #                                       for t in self.targets_id]):
        #     pc_targets_id = []
        #     for tar_id in self.targets_id:
        #         pc_tar_id = 'PC_' + tar_id
        #         pc_targets_id.append(pc_tar_id)
        #         self.targets[pc_tar_id] = self.targets[tar_id]
        #     self.targets_id += pc_targets_id
        
        # TODO Dynamically gets info: xna_region? focus_len?
        
        # This values will work only for the short version
        # TODO version for long ref. Add left/right arm len? trim tar inside loop?
        
        self.barcodes = {}
        self.barcodes_pos = {}
        self.barcodes_pos_rev = {}
        self.xna_kmers = {}
        self.xna_kmers_pos = {}
        self.xna_kmers_pos_rev = {}
        self.xna_kmers_len = {}
        self.ub_pos = self.x_pos = {}
        self.x_pos_rev = {}
        self.x_pos_xna_kmers = {}
        
        self.ub_area_seq = {}
        self.ub_area_seq_per_pos = {}
        self.len_targets = {}
        
        if short_version:
            bc_slice = slice(self.left_primer_len, 
                             self.left_primer_len + self.barcode_len)
            kmer_slice = slice(
                self.left_primer_len + self.barcode_len + self.middle_primer_len, 
                -self.right_primer_len)
        else:
            bc_slice = slice(self.left_arm_len + self.left_primer_len, 
                   self.left_arm_len + self.left_primer_len + self.barcode_len)
            kmer_slice = slice(self.left_arm_len +
                self.left_primer_len + self.barcode_len + self.middle_primer_len, 
                - self.right_primer_len - self.right_arm_len)
        
        #### Extracting info per target/template
        for tar_id, tar in self.targets.items():
            self.len_targets[tar_id] = len(tar)
            self.barcodes[tar_id] = tar[bc_slice]
            self.xna_kmers[tar_id] = tar[kmer_slice]
            self.xna_kmers_len[tar_id] = len(tar[kmer_slice])
            self.x_pos[tar_id] = [_.start() for _ in re.finditer('N', tar)]
            
            self.x_pos_rev[tar_id] = [ len(tar) - p - 1 for p in self.x_pos[tar_id][::-1] ]
            self.x_pos_xna_kmers[tar_id] = [_.start() for _ in 
                                            re.finditer('N', tar[kmer_slice])]
            
            self.barcodes_pos[tar_id] = (bc_slice.start, bc_slice.stop)
            self.barcodes_pos_rev[tar_id] = tuple([ len(tar) - p - 1 for p in 
                                              self.barcodes_pos[tar_id][::-1] ])
            self.xna_kmers_pos[tar_id] = (kmer_slice.start, 
                                          kmer_slice.start + len(tar[kmer_slice]))
            self.xna_kmers_pos_rev[tar_id] = [ len(tar) - p - 1 for p in 
                                              self.xna_kmers_pos[tar_id][::-1] ]
            
            
            x_pos = self.x_pos[tar_id]
            if len(x_pos) > 0:
                self.ub_area_seq[tar_id] = tar[x_pos[0]-5: x_pos[-1]+6]
                # self.ub_area_seq_per_pos[tar_id] = [ tar[pos-5: pos+6] 
                #                                      for pos in x_pos ]
                self.ub_area_seq_per_pos[tar_id] = { pos: tar[pos-5: pos+6] 
                                                     for pos in x_pos }
            
            
            if tar_id == 'PC15' and ref_name == 'XNA16':
                self.barcodes[tar_id] = tar[bc_slice.start-1:bc_slice.stop-1]
                self.xna_kmers[tar_id] = tar[kmer_slice.start-1:kmer_slice.stop]
                self.xna_kmers_len[tar_id] = len(self.xna_kmers[tar_id])
                # self.barcodes[tar_id] = 'xx'
                self.barcodes_pos[tar_id] = (bc_slice.start-1, bc_slice.stop-1)
                self.xna_kmers_pos[tar_id] = (kmer_slice.start-1, 
                          kmer_slice.start-1 + len(self.xna_kmers[tar_id]))
                ### No need to update _pos_rev because for reverse del is after barcode?
            
            # if not tar_id.startswith('PC'):
            #     self.x_pos[tar_id] = [_.start() for _ in re.finditer('N', tar)]
            # else:
            #     xna_tar_id = next(i for i in self.targets_id 
            #           if i.endswith(tar_id[2:]) and not i.startswith('PC') )
            #     xna_tar = self.targets[xna_tar_id]
            #     self.x_pos[tar_id] = [_.start() for _ in re.finditer('N', xna_tar)]
        
        all_bcs = list(self.barcodes.values())
        self.barcodes_cnt = { tar:all_bcs.count(bc) for tar,bc in self.barcodes.items() }
        
    
    def locate_read(self, barcode_start, barcode_end, target_id, strand, length):
        read_start = barcode_start - self.left_primer_len
        read_end = (barcode_end + self.middle_primer_len +
                    self.xna_kmers_len[target_id] +
                    self.right_primer_len)
        
        if target_id == 'PC15' and self.ref_name == 'XNA16':
            # Exception case with smaller left_primer
            read_start -= 1
        
        if strand == 'R':
            new_read_start = length - read_end
            read_end = length - read_start
            read_start = new_read_start
        
        return read_start, read_end
    
    def get_complement_target_id(self, target_id, use_aliases=False):
        if self.ref_name.startswith('XNA1024'):
            return target_id
        
        # suffix_len = (3 if self.ref_name in ['XNA_4DS'] else 2)
        suffix_len = (3 if self.ref_name in ['XNA_4Ds'] else 2)
        
        if not use_aliases:
            targets_id = self.targets_id
        else:
            if target_id not in self.aliases_targets_id:
                target_id = self.aliases[target_id]
            targets_id = self.aliases_targets_id
            suffix_len = 2
        
        if target_id.startswith('PC'):
            cpl_target_id = next(i for i in targets_id 
                  if i.endswith(target_id[suffix_len:]) and not i.startswith('PC') )
        else:
            pc_targets_id = [i for i in targets_id if i.startswith('PC')]
            cpl_target_id = next(i for i in pc_targets_id 
                                 if target_id.endswith(i[suffix_len:]))
        
        return cpl_target_id
    
    def pretty_print_target(self, target_id, join_str=' ', print_header=False):
        target = self.targets[target_id]
        lef_primer = target[:self.barcodes_pos[target_id][0]]
        barcode = self.barcodes[target_id]
        middle_primer = target[self.barcodes_pos[target_id][1]:self.xna_kmers_pos[target_id][0]]
        kmer = self.xna_kmers[target_id]
        right_primer = target[self.xna_kmers_pos[target_id][1]:]
        
        if print_header:
            print(join_str.join(['lef_primer','barcode','middle_primer','UB_kmer','right_primer']))
        
        print(join_str.join([lef_primer,barcode,middle_primer,kmer,right_primer]))
        # print(''.join([lef_primer.lower(),barcode,middle_primer.lower(),kmer,right_primer.lower()]))
    
    def get_ub_area(self, target_id, ub_order, reverse=False, kmer_len=6, replace_N=True):
        if not (0 < ub_order <= len(self.x_pos[target_id])):
            raise ValueError(f"UB order out of range: 1 <= {ub_order} <= {len(self.x_pos[target_id])}")
        if not reverse:
            x_pos = self.x_pos[target_id][ub_order-1]
        else:
            x_pos = self.x_pos[target_id][::-1][ub_order-1]
        ub_area = self.targets[target_id][x_pos-kmer_len+1:x_pos+kmer_len]
        
        if replace_N:
            ub_area = ub_area.replace('N','X')
        if reverse:
            ub_area = reverse_complement(ub_area)
        
        return ub_area
    
    def get_ub_kmers(self, xna_target_id, x_pos=None, reverse=False, kmer_len=6, 
                     flat=False, return_pos=False):
        if x_pos is None:
            tar_ub_kmers = [ 
                self.get_ub_kmers(xna_target_id, x_pos=x_pos, reverse=reverse, 
                                  kmer_len=kmer_len, return_pos=False)
                for x_pos in self.x_pos[xna_target_id]]
            pos_ub_kmers = []
            for x_pos in self.x_pos[xna_target_id]:
                pos_ub_kmers.append(list(range(x_pos-kmer_len+1, x_pos+1)))
        else:
            xna_target = self.targets[xna_target_id]
            ub_kmer_range = xna_target[x_pos-kmer_len+1:x_pos+kmer_len]
            # ub_kmer_range = ub_kmer_range.replace('N', 'X')
            
            tar_ub_kmers = sliding_window_view(list(ub_kmer_range), kmer_len)
            tar_ub_kmers = [ ''.join(k) for k in tar_ub_kmers ]
            
            pos_ub_kmers = list(range(x_pos-kmer_len+1, x_pos+1))
            
            if reverse:
                tar_ub_kmers = [ reverse_complement(k) for k in tar_ub_kmers[::-1] ]
                pos_ub_kmers = [ len(xna_target) - p - 1 for p in pos_ub_kmers[::-1] ]
        
        if flat:
            tar_ub_kmers = [ item for list in tar_ub_kmers for item in list ]
            pos_ub_kmers = [ item for list in pos_ub_kmers for item in list ]
            # tar_ub_kmers = list(dict.fromkeys(tar_ub_kmers)) # To remove dups (XNA13)
            # pos_ub_kmers = list(dict.fromkeys(pos_ub_kmers))
        
        if not return_pos:
            ret = tar_ub_kmers
        else:
            ret = [tar_ub_kmers, pos_ub_kmers]
        return ret
        
    def search_kmer(self, ref_kmer, strands=['F','R']):
        target_ids = []
        
        for target_id, target in self.targets.items():
            if 'F' in strands and ref_kmer in target:
                target_ids.append(target_id)
                continue
            
            if 'R' in strands and ref_kmer in reverse_complement(target):
                target_ids.append(target_id)
        
        return target_ids
        
def identify_ref(targets_ids):
    ### Func that identify correct ref name by template id, and return it
    for ref_name in VALID_REFS:
        refs_info = XNA_refs(ref_name)
        
        # Check if has intersection with targt ids
        inter = set(refs_info.targets_id).intersection(targets_ids)
        if ref_name == 'XNA_4Ds' and len(inter) == 0:
            inter = set(refs_info.aliases_targets_id).intersection(targets_ids)
        
        if len(inter) > 0:
            break
        
        refs_info = None
    return refs_info
    