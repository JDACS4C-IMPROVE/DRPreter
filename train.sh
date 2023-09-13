#!/bin/bash
  
#########################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE #PATH# WITH THE MODEL EXECUTABLE.
#########################################################################


# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###
CANDLE_MODEL=/usr/local/DRPreter/train.py
DATA_PREPROCESSOR=/usr/local/DRPreter/preprocess.sh

if [ $# -lt 2 ] ; then
        echo "Illegal number of parameters"
        echo "CUDA_VISIBLE_DEVICES and CANDLE_DATA_DIR are required"
        exit -1
fi

if [ $# -eq 2 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift
        CMD="python ${CANDLE_MODEL}"
        echo "CMD = $CMD"

elif [ $# -ge 3 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift

        # if original $3 is a file, set candle_config and passthrough $@
        if [ -f $CANDLE_DATA_DIR/$1 ] ; then
		echo "$1 is a file"
                CANDLE_CONFIG=$1 ; shift
                CMD="python ${CANDLE_MODEL} --config_file $CANDLE_CONFIG $@"
                echo "CMD = $CMD $@"

        # else passthrough $@
        else
		echo "$1 is not a file"
                CMD="python ${CANDLE_MODEL} $@"
                echo "CMD = $CMD"

        fi
fi

# Set data preprocessing command
DPP_CMD="python ${DATA_PREPROCESSOR}"

# Display runtime arguments
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"

# Set up environmental variables and execute model
EXE_DIR=$(dirname ${CANDLE_MODEL})
cd $EXE_DIR

echo "running command ${DPP_CMD}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $DPP_CMD
echo "running command ${CMD}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD

# Change into directory and execute tests
# cd ${MODEL_DIR}


# model data processing and loading
# python HiDRA_FeatureGeneration.py

# model training
# python HiDRA_training.py

# Check if successful
exit 0
