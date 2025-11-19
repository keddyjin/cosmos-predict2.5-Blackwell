#!/bin/bash
set +e

# Set the output directory
OUTPUT_DIR="outputs/predict_2.5_baseline"
mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=8

INFERENCE_SCRIPT="examples/inference.py"
MULTIVIEW_SCRIPT="examples/multiview.py"
ACTION_CONDITIONED_SCRIPT="examples/action_conditioned.py"

# Base/T2W
echo "============== Base/T2W: 1 GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/base_text2world_1GPU
mkdir -p $CUR_OUTPUT_DIR
python $INFERENCE_SCRIPT -i assets/base/robot_pouring.json -o $CUR_OUTPUT_DIR --inference-type=text2world 2>&1 | tee $CUR_OUTPUT_DIR/log
echo "============== Base/T2W: $NUM_GPUS GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/base_text2world_${NUM_GPUS}GPU
mkdir -p $CUR_OUTPUT_DIR
torchrun --nproc_per_node=$NUM_GPUS $INFERENCE_SCRIPT -i assets/base/robot_pouring.json -o $CUR_OUTPUT_DIR --inference-type=text2world 2>&1 | tee $CUR_OUTPUT_DIR/log

# Base/I2W
echo "============== Base/I2W: 1 GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/base_image2world_1GPU
mkdir -p $CUR_OUTPUT_DIR
python $INFERENCE_SCRIPT -i assets/base/robot_pouring.json -o $CUR_OUTPUT_DIR --inference-type=image2world 2>&1 | tee $CUR_OUTPUT_DIR/log
echo "============== Base/I2W: $NUM_GPUS GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/base_image2world_${NUM_GPUS}GPU
mkdir -p $CUR_OUTPUT_DIR
torchrun --nproc_per_node=$NUM_GPUS $INFERENCE_SCRIPT -i assets/base/robot_pouring.json -o $CUR_OUTPUT_DIR --inference-type=image2world 2>&1 | tee $CUR_OUTPUT_DIR/log

# Base/V2W
echo "============== Base/V2W: 1 GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/base_video2world_1GPU
mkdir -p $CUR_OUTPUT_DIR
python $INFERENCE_SCRIPT -i assets/base/robot_pouring.json -o $CUR_OUTPUT_DIR --inference-type=video2world 2>&1 | tee $CUR_OUTPUT_DIR/log
echo "============== Base/V2W: $NUM_GPUS GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/base_video2world_${NUM_GPUS}GPU
mkdir -p $CUR_OUTPUT_DIR
torchrun --nproc_per_node=$NUM_GPUS $INFERENCE_SCRIPT -i assets/base/robot_pouring.json -o $CUR_OUTPUT_DIR --inference-type=video2world 2>&1 | tee $CUR_OUTPUT_DIR/log

# auto/T2W
echo "============== auto/T2W: $NUM_GPUS GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/multiview_text2world_${NUM_GPUS}GPU
mkdir -p $CUR_OUTPUT_DIR
torchrun --nproc_per_node=$NUM_GPUS $MULTIVIEW_SCRIPT -i assets/multiview/urban_freeway.json -o $CUR_OUTPUT_DIR --inference-type=text2world 2>&1 | tee $CUR_OUTPUT_DIR/log

# auto/I2W
echo "============== auto/I2W: $NUM_GPUS GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/multiview_image2world_${NUM_GPUS}GPU
mkdir -p $CUR_OUTPUT_DIR
torchrun --nproc_per_node=$NUM_GPUS $MULTIVIEW_SCRIPT -i assets/multiview/urban_freeway.json -o $CUR_OUTPUT_DIR --inference-type=image2world 2>&1 | tee $CUR_OUTPUT_DIR/log

# auto/V2W
echo "============== auto/V2W: $NUM_GPUS GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/multiview_video2world_${NUM_GPUS}GPU
mkdir -p $CUR_OUTPUT_DIR
torchrun --nproc_per_node=$NUM_GPUS $MULTIVIEW_SCRIPT -i assets/multiview/urban_freeway.json -o $CUR_OUTPUT_DIR --inference-type=video2world 2>&1 | tee $CUR_OUTPUT_DIR/log

# robot
echo "============== robot: 1 GPUs =============="
CUR_OUTPUT_DIR=$OUTPUT_DIR/action_conditioned_basic_1GPU
mkdir -p $CUR_OUTPUT_DIR
python $ACTION_CONDITIONED_SCRIPT -i assets/action_conditioned/basic/inference_params.json -o $CUR_OUTPUT_DIR 2>&1 | tee $CUR_OUTPUT_DIR/log
