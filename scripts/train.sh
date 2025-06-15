#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
GPU=None
OID="None"
LABEL="None"


while getopts "p:d:c:n:w:g:r:o:l:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    o)
      OID=$OPTARG
      ;;
    l)
      LABEL=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts launch pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
# export PYTHONPATH=./$CODE_DIR:/usr/local/lib/python3.8/dist-packages
export PYTHONPATH=./$CODE_DIR:/opt/conda/envs/part/lib/python3.10/site-packages
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="
# $PYTHON "$CODE_DIR"/launch/$TRAIN_CODE \
#     --config-file "$CONFIG_DIR" \
#     --num-gpus "$GPU" \
#     --options save_path="$EXP_DIR" weight="$LAST_WEIGHT" oid="$OID" label="$LABEL"
export WEIGHT=None

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/launch/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" oid="$OID" label="$LABEL"
else
    $PYTHON "$CODE_DIR"/launch/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" weight="$WEIGHT" oid="$OID" label="$LABEL"
fi