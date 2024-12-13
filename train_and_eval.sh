#!/bin/bash
set -e ## For exiting on first error encountered

echo "> Starting Train and Eval Bonito Model" - `date`
printf "\n"

#### Script command line arguments ###########################################

if [ $# -lt 3 ]; then
    >&2 echo "[ERROR] Missing arguments"
    >&2 echo "Usage: $(basename $0) EVAL_EXP TRAIN_DIR CTC_DIR [-b BATCHSIZE] [-u UBS] [-e EPOCHS] [-f|freeze] ..."
    exit 1
fi

EVAL_EXP=${1}
TRAIN_DIR=${2}
DATA_TRAIN_DIR=${3}
PROP_UBS=${4}
OPTIND=5

### Default arguments
EXTRA_EVAL_EXPS=''
EPOCHS=5
PRETRAINED_MODEL='xna_r9.4.1_e8_sup@v3.3'
LR='5e-4' # 2e-3 > [5e-4] > 5e-5
DR=0.50
DR_BOTTOM=0.05
NOISE_STD=1.00 # 2.00
STD_DIST='--std-dist truncnorm_shift_1.5_0.5'
SPIKE='--spike' # Assuming spike/hybrid as default, unless STITCH_MODE is set

### Default arguments Stitch aug
# STITCH_MODE='--stitch-mode per_kmer'
CAND_SAMPLE_SIZE='--cand-sample-size 5'

while getopts "h?b:s:e:l:d:D:p:fF:n:u:r:v:S:E:m:c:w:WZX:P:N:M:U:B:" opt; do
  case "$opt" in
    h|\?)
      echo "Usage: $(basename $0) EVAL_EXP TRAIN_DIR CTC_DIR PROP_UBS [-b BATCHSIZE] [-u UBS] [-s STRAND] [-e EPOCHS] [-l LR] [-d DR] [-D DR_BOTTOM] [-f|freeze] [-F NUM_UNFREEZE_TOP] [-p PRETRAINED_MODEL] [-n NOISE_STD] [-r ref-kmer-filepath] [-v VAR_PROP_UBS] [-S STD_DIST] [-E EXTRA_EVAL_EXPS] [-c CAND_SAMPLE_SIZE] ..."
      exit 1
      ;;
    b)  BATCHSIZE=$OPTARG ;;
    e)  EPOCHS=$OPTARG ;;
    l)  LR=$OPTARG ;;
    d)  DR=$OPTARG ;;
    D)  DR_BOTTOM=$OPTARG ;;
    p)  PRETRAINED_MODEL=$OPTARG ;;
    f)  FREEZE=' --freeze-bottom ' ;;
    F)  NUM_UNFREEZE_TOP=' --num-unfreeze-top '$OPTARG ;;
    u)  UBS='--ubs '$OPTARG
		case "$OPTARG" in
			X) STRAND=F ;;
			Y) STRAND=R ;;
		esac ;;
    s)  STRAND=$OPTARG ;; # Better to use -u ubs only
    v)  VAR_PROP_UBS='--var-prop-ubs '$OPTARG ;;
    B)  UB_PAD='--ub-pad '$OPTARG ;;
    E)  EXTRA_EVAL_EXPS=$OPTARG ;;
	
	### Spiking args
    r)  REF_KMER='--ref-filepath '$OPTARG ;;
    n)  NOISE_STD=$OPTARG ;;
    S)  STD_DIST='--std-dist '$OPTARG ;;
    Z)  FULLY_SYNTH='--fully_synth ';; # Hybrid otherwise
    U)  SYNTH_PROP_UBS='--synth-prop-ubs '$OPTARG ;; # In case of mixing synth+spliced
	
	### Stitching args
    m)  STITCH_MODE='--stitch-mode '$OPTARG 
		SPIKE='' ;;
    X)  XNA_DIR='--xna_ctc_dir '$OPTARG;;
    c)  CAND_SAMPLE_SIZE='--cand-sample-size '$OPTARG ;;
    W)  WEIGHTED_POS_PICK='--weighted-pos-pick ';;
    N)  STITCH_NOISE_STD='--stitch-noise-std '$OPTARG ;;
    M)  STITCH_NOISE_MODE='--stitch-noise-mode '$OPTARG ;;
    P)  PERMUTE_WIN_SIZE='--permute-win-size '$OPTARG ;;
	
    w)  NUM_WORKERS='--num-workers '$OPTARG ;;
  esac
done

if [ -n "$SYNTH_PROP_UBS" ]; then
	SPIKE='--spike'
fi

echo "Arguments:"
echo "EVAL_EXP="$EVAL_EXP
echo "EXTRA_EVAL_EXPS="$EXTRA_EVAL_EXPS
echo "TRAIN_DIR="$TRAIN_DIR
echo "DATA_TRAIN_DIR="$DATA_TRAIN_DIR
echo "UBS="$UBS" | STRAND="$STRAND
echo "PROP_UBS="$PROP_UBS" | VAR_PROP_UBS="$VAR_PROP_UBS" | SYNTH_PROP_UBS="$SYNTH_PROP_UBS" | UB_PAD="$UB_PAD
echo "SPIKE="$SPIKE" | REF_KMER="$REF_KMER" | FULLY_SYNTH="$FULLY_SYNTH
echo "STD_DIST="$STD_DIST" | NOISE_STD="$NOISE_STD
echo "PRETRAINED_MODEL="$PRETRAINED_MODEL
echo "BATCHSIZE="$BATCHSIZE" | EPOCHS="$EPOCHS" | LR="$LR
echo "DR="$DR" | DR_BOTTOM="$DR_BOTTOM
echo "FREEZE="$FREEZE" | NUM_UNFREEZE_TOP="$NUM_UNFREEZE_TOP
echo "STITCH_MODE="$STITCH_MODE" | CAND_SAMPLE_SIZE="$CAND_SAMPLE_SIZE" | WEIGHTED_POS_PICK="$WEIGHTED_POS_PICK
# echo "PERMUTE_WIN_SIZE="$PERMUTE_WIN_SIZE" | STITCH_NOISE_STD="$STITCH_NOISE_STD
echo "STITCH_NOISE_MODE="$STITCH_NOISE_MODE" | STITCH_NOISE_STD="$STITCH_NOISE_STD

##### 1) Train Model ###########################################
printf "\n+++++ 1) Train Model ++++++++++++++++++++++++++++++++++++++++\n"


if [[ ! -e $TRAIN_DIR/training.csv || ! -s $TRAIN_DIR/training.csv ]]; then # Not exist or empty
	if [[ -e $TRAIN_DIR/training.csv && ! -s $TRAIN_DIR/training.csv ]]; then # Exist and empty
		echo "[WARNING] empty training file, removing dir before training."
		rm -fr -s $TRAIN_DIR/;
	fi
	
	source ./ub-bonito/venv3/bin/activate;
	
	if [ -n "$BATCHSIZE" ]; then
		TRAIN_BS='--batch '$BATCHSIZE
	fi
	
	(set -x; 
	bonito train --epochs $EPOCHS --lr $LR \
		--drop-rate $DR --drop-rate-bottom $DR_BOTTOM \
		$FREEZE $NUM_UNFREEZE_TOP \
		$UBS --prop-ubs $PROP_UBS $VAR_PROP_UBS \
		$SPIKE --noise-std $NOISE_STD --variable-noise \
		$STD_DIST $REF_KMER $FULLY_SYNTH \
		$STITCH_MODE $CAND_SAMPLE_SIZE $WEIGHTED_POS_PICK $XNA_DIR \
		$PERMUTE_WIN_SIZE $STITCH_NOISE_STD $STITCH_NOISE_MODE \
		$SYNTH_PROP_UBS \
		$UB_PAD \
		$NUM_WORKERS \
		--pretrained $PRETRAINED_MODEL --directory $DATA_TRAIN_DIR \
		$TRAIN_BS \
		$TRAIN_DIR);
	
	deactivate;
else
	echo "Skipping training..."
fi

##### 1.5) run_ub_validation.sh ###########################################
printf "\n+++++ 1.5) run_ub_validation.sh ++++++++++++++++++++++++++++++++++++++++\n"

if [ -n "$BATCHSIZE" ]; then
	EXTRA_ARGS='-b '$BATCHSIZE
fi
if [ -n "$STRAND" ]; then
	EXTRA_ARGS=$EXTRA_ARGS' -s '$STRAND
fi
(set -x; ./run_ub_validation.sh $EVAL_EXP $TRAIN_DIR $EXTRA_ARGS)

##### 2) eval_model.sh ###########################################
printf "\n+++++ 2) eval_model.sh ++++++++++++++++++++++++++++++++++++++++\n"

if [[ -e $TRAIN_DIR/training.csv && -s $TRAIN_DIR/training.csv ]]; then # Exist and not empty
	(set -x; ./eval_model.sh $EVAL_EXP $TRAIN_DIR $EXTRA_ARGS);
	
	for EXTRA_EXP in $EXTRA_EVAL_EXPS; do
		(set -x; ./eval_model.sh $EXTRA_EXP $TRAIN_DIR $EXTRA_ARGS);
	done;
else
	echo "> Invalid training dir"
	echo "Skipping eval model..."
fi

printf "\n"
echo "> Train and Eval Bonito Model finished " - `date`