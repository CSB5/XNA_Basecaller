#!/bin/bash
set -e ## For exiting on first error encountered

echo "> Starting model evaluation" - `date`
printf "\n"

#### Script command line arguments ###########################################

if [ $# -lt 2 ]; then
    >&2 echo "[ERROR] Missing arguments"
    >&2 echo "Usage: $(basename $0) EXP TRAIN_DIR [-u UBS] [-b BATCHSIZE] [-S SPLIT] [-w WEIGHTS]"
    exit 1
fi

EXP=${1} # POC | CPLX
TRAIN_DIR=${2}
OPTIND=3

### Default arguments
BASECALLS_DIR=$TRAIN_DIR/'basecalls'
SPLIT='test'

while getopts "h?b:s:u:w:S:" opt; do
  case "$opt" in
    h|\?)
      echo "Usage: $(basename $0) EXP TRAIN_DIR [-u UBS] [-b BATCHSIZE] [-S SPLIT] [-w WEIGHTS]"
      exit 1
      ;;
    b)  BATCHSIZE='--batch '$OPTARG ;;
    s)  STRAND=$OPTARG ;;
    S)  SPLIT=$OPTARG ;;
	u)  case "$OPTARG" in
			X) STRAND=F ;;
			Y) STRAND=R ;;
		esac ;;
    w)  WEIGHTS='--weights '$OPTARG
		BASECALLS_DIR=$TRAIN_DIR/'basecalls-weights_'$OPTARG
		;;
  esac
done

echo "Arguments:"
echo "EXP="$EXP" | SPLIT="$SPLIT" | STRAND="$STRAND
echo "TRAIN_DIR="$TRAIN_DIR
echo "BATCHSIZE="$BATCHSIZE
echo "BASECALLS_DIR="$BASECALLS_DIR
echo "WEIGHTS="$WEIGHTS

#### Hard-coded Params ###########################################

REF_FILE=refdb_short.fasta

if [[ $EXP == 'POC' || $EXP == 'poc' ]]; then
	REF_NAME='XNA20'; EXP='POC';
elif [[ $EXP == 'CPLX' || $EXP == 'cplx' ]]; then
	REF_NAME='XNA1024'; EXP='CPLX';
else 
	echo "Unknown experiment: "$EXP
	exit 1
fi
SAMPLE=$EXP-$SPLIT
STRAND_LIST='split_reads-'$SPLIT'.tsv';

if [[ $REF_NAME == 'XNA16' || $REF_NAME == 'XNA_4Ds' || $REF_NAME == 'XNA20' ]]; then
	MAX_BC_DIST=5;
elif [[ $REF_NAME == 'XNA1024' || $REF_NAME == 'XNA1024-A027' ]]; then
	MAX_BC_DIST=8;
fi

# EXP_DIR=$HOME'/projects/xna_basecallers/exps/'$EXP
# REFS_DIR=$HOME'/projects/xna_basecallers/xna_refs/'$REF_NAME
EXP_DIR='./xna_libs/'$EXP
REFS_DIR=$EXP_DIR

if [[ -n $STRAND ]]; then
	BASE_STRAND_LIST=$(echo $STRAND_LIST | cut -d. -f1)
	STRAND_LIST=$BASE_STRAND_LIST'-strands_'$STRAND'.tsv'
fi

echo "REF_NAME="$REF_NAME" | SAMPLE="$SAMPLE" | STRAND_LIST="$STRAND_LIST
# printf "\n"
# echo "Hard-coded Params:"
# echo "REFS_DIR="$REFS_DIR
# echo "EXP_DIR="$EXP_DIR
# echo "MAX_BC_DIST="$MAX_BC_DIST

if [[ $TRAIN_DIR == 'help' ]]; then
	exit
fi

##### 1) basecalling ###########################################
printf "\n+++++ 1) basecalling ++++++++++++++++++++++++++++++++++++++++\n"

BASECALLS_FILE='reads-'$SAMPLE'.fastq'
# echo "BASECALLS_FILE="$BASECALLS_FILE

if [[ ! -e $BASECALLS_DIR/$BASECALLS_FILE || ! -s $BASECALLS_DIR/$BASECALLS_FILE ]]; then
	echo "Basecalls filepath: "$BASECALLS_DIR/$BASECALLS_FILE;
	mkdir -p $BASECALLS_DIR;
	source ./ub-bonito/venv3/bin/activate;
		
	(set -x; # For displaying the command before executing it.
	bonito basecaller $TRAIN_DIR $EXP_DIR/reads -v \
		--read-ids $EXP_DIR/$STRAND_LIST \
		$BATCHSIZE $WEIGHTS \
		> $BASECALLS_DIR/$BASECALLS_FILE);
	
	deactivate;
	
	if [[ ! -s $BASECALLS_DIR/$BASECALLS_FILE ]]; then
		echo "[ERROR] basecalls file is empty!";
		rm -f $BASECALLS_DIR/$BASECALLS_FILE;
	fi
else
	echo -e "Fastq file found: "$BASECALLS_FILE;
	echo "Skipping..."
fi

##### 2) minimap2 ###########################################
printf "\n+++++ 2) minimap2 ++++++++++++++++++++++++++++++++++++++++\n"

ALIGN_FILE_PREFIX='alignment-'$SAMPLE
ALIGN_FILE=$ALIGN_FILE_PREFIX'.paf'

# if true; then
if [[ -e $BASECALLS_DIR/$BASECALLS_FILE && ! -e $BASECALLS_DIR/$ALIGN_FILE.gz && ! -e $BASECALLS_DIR/$ALIGN_FILE ]]; then
	(set -x;
	bin/minimap2 \
		-x map-ont -t 12 -w 5 -c --cs=short --secondary=no \
		$REFS_DIR/$REF_FILE \
		$BASECALLS_DIR/$BASECALLS_FILE \
		-o $BASECALLS_DIR/$ALIGN_FILE);
	
	if [[ -s $BASECALLS_DIR/$ALIGN_FILE ]]; then
		# gzip alignment.paf;
		echo "Alignment finished."
	else
		rm -f $BASECALLS_DIR/$ALIGN_FILE;
	fi
else
	if [[ ! -e $BASECALLS_DIR/$BASECALLS_FILE ]]; then
		echo "[ERROR] Basecalls NOT found: "$BASECALLS_DIR/$BASECALLS_FILE;
	elif [[ -e $BASECALLS_DIR/$ALIGN_FILE.gz || -e $BASECALLS_DIR/$ALIGN_FILE ]]; then
		echo "Alignment file found: "$ALIGN_FILE;
	fi
	
	echo "Skipping..."
fi

##### 3) analyze_paf.py ###########################################
printf "\n+++++ 3) analyze_paf.py ++++++++++++++++++++++++++++++++++++++++\n"

# if false; then
if [[ -e $BASECALLS_DIR/$ALIGN_FILE && -s $BASECALLS_DIR/$ALIGN_FILE ]]; then
	RES_FILE=results_summ-$SAMPLE.csv
	if [[ ! -e $BASECALLS_DIR/$RES_FILE ]]; then
		if [ -n "$STRAND" ]; then
			EXTRA_ARGS='-S '$STRAND
		fi
		(set -x;
		python src/tools/analyze_paf.py $EXP $BASECALLS_DIR/$ALIGN_FILE \
			-p -D -d $MAX_BC_DIST $EXTRA_ARGS \
			-R $BASECALLS_DIR/$BASECALLS_FILE);
	else
		echo -e "Results file found: "$RES_FILE;
		echo -e "Skipping...";
	fi
else
	if [[ -s $BASECALLS_DIR/$ALIGN_FILE ]]; then
		echo "[ERROR] Alignment file NOT found: "$BASECALLS_DIR/$ALIGN_FILE;
	else
		echo "[WARNING] Alignment file is empty! No reads could be aligned.";
		echo "> "$BASECALLS_DIR/$ALIGN_FILE;
	fi
	
	echo "Skipping..."
fi

printf "\n"
echo "> Model evaluation finished" - `date`
