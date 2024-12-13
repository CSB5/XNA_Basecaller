#!/bin/bash
set -e ## For exiting on first error encountered

# Execute basecallings at validation sampling and select epoch with best UB Acc.

echo "> Starting run_ub_validation" - `date`
printf "\n"

#### Script command line arguments ###########################################

if [ $# -lt 2 ]; then
    >&2 echo "[ERROR] Missing arguments"
    >&2 echo "Usage: $(basename $0) TRAIN_DIR EXP [-b BATCHSIZE] [-s STRAND]"
    exit 1
fi

EXP=${1} # A003 | A007+A008 | A026+A027
TRAIN_DIR=${2}
OPTIND=3

while getopts "h?b:s:" opt; do
  case "$opt" in
    h|\?)
      echo "Usage: $(basename $0) TRAIN_DIR EXP [-b BATCHSIZE] [-s STRAND]"
      exit 1
      ;;
    b)  BATCHSIZE=$OPTARG
      ;;
    s)  STRAND=$OPTARG
      ;;
  esac
done

echo "Arguments:"
echo "TRAIN_DIR="$TRAIN_DIR
echo "EXP="$EXP" | STRAND="$STRAND
echo "BATCHSIZE="$BATCHSIZE

#### Hard-coded Params ###########################################

if [[ $EXP == 'POC' ]]; then
	SAMPLE='POC-val';
elif [[ $EXP == 'CPLX' ]]; then
	SAMPLE='CPLX-val';
else 
	echo "Unknown experiment: "$EXP
	exit 1
fi

printf "\n"
echo "Hard-coded Params:"
echo "SAMPLE="$SAMPLE

##### 1) eval_model-val-weights.sh ###########################################
printf "\n+++++ 1) eval_model.sh validation ++++++++++++++++++++++++++++++++++++++++\n"

if [ -n "$BATCHSIZE" ]; then
	EXTRA_ARGS='-b '$BATCHSIZE
fi
if [ -n "$STRAND" ]; then
	EXTRA_ARGS=$EXTRA_ARGS' -s '$STRAND
fi

# for WEIGHTS_FILE in $(ls $TRAIN_DIR/weights_*.tar); do
for WEIGHTS_FILE in $(find $TRAIN_DIR/weights_*.tar -type f); do
	W=$(echo $WEIGHTS_FILE | grep -oP 'weights_\K\d+')
	
	RES_FILE=results_summ-$SAMPLE.csv
	if [[ ! -e $TRAIN_DIR/basecalls-weights_$W/$RES_FILE ]]; then
		# (set -x; ./eval_model-val-weights.sh $TRAIN_DIR $EXP $W $BATCHSIZE $STRAND);
		(set -x; ./eval_model.sh $EXP $TRAIN_DIR -w $W -S val $EXTRA_ARGS);
	else
		echo -e "\t [Skipping] Results file found: "basecalls-weights_$W/$RES_FILE;
	fi
done

##### 2) consolidate_ub_validation.py ###########################################
printf "\n+++++ 2) consolidate_ub_validation.py ++++++++++++++++++++++++++++++++++++++++\n"

(set -x; 
python src/tools/consolidate_ub_validation.py $TRAIN_DIR -t $SAMPLE -s);

printf "\n"
echo "> run_ub_validation finished " - `date`