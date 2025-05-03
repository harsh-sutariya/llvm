#!/bin/bash

# Define default values
MODEL_PATH="/scratch/spp9399/mia/llava_lora_our_loss"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
DATA_PATH="/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/sequence_test_images.json"
IMAGE_DIR="/scratch/spp9399/MIA-DPO/gen_data/gen_instruction"
OUTPUT_PATH="output_json.json"
USE_ADAPTVIS=true
PRECISION="fp16"
TEMPERATURE=0.2
TOP_P=0.95
MAX_NEW_TOKENS=1024
CONF_LOW=0.3
CONF_HIGH=0.7
ALPHA_LOW=0.5
ALPHA_HIGH=2.0
IMAGE_ASPECT_RATIO="pad"
TEST_MODE=false
TEST_SIZE=3
TEST_MAX_TOKENS=100
VERBOSE=false

export cache_dir=/scratch/hs5580/mia2/cache

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-base)
      MODEL_BASE="$2"
      shift 2
      ;;
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --use-adaptvis)
      USE_ADAPTVIS=true
      shift
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --conf-low)
      CONF_LOW="$2"
      shift 2
      ;;
    --conf-high)
      CONF_HIGH="$2"
      shift 2
      ;;
    --alpha-low)
      ALPHA_LOW="$2"
      shift 2
      ;;
    --alpha-high)
      ALPHA_HIGH="$2"
      shift 2
      ;;
    --image-aspect-ratio)
      IMAGE_ASPECT_RATIO="$2"
      shift 2
      ;;
    --test)
      TEST_MODE=true
      shift
      ;;
    --test-size)
      TEST_SIZE="$2"
      shift 2
      ;;
    --test-max-tokens)
      TEST_MAX_TOKENS="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [[ -z "$MODEL_PATH" && -z "$MODEL_TYPE" ]]; then
  echo "Error: Either --model-path or --model must be specified"
  exit 1
fi

if [[ -z "$DATA_PATH" ]]; then
  echo "Error: --data must be specified"
  exit 1
fi

if [[ -z "$IMAGE_DIR" ]]; then
  echo "Error: --image-dir must be specified"
  exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  echo "Error: --output must be specified"
  exit 1
fi

# Set model path based on model type if specified
if [[ -n "$MODEL_TYPE" && -z "$MODEL_PATH" ]]; then
  case "$MODEL_TYPE" in
    llava)
      MODEL_PATH="/scratch/spp9399/models/llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234"
      ;;
    llava_dpo)
      # Path to regular DPO-tuned LLaVA model
      MODEL_PATH="/path/to/llava_dpo_model"
      ;;
    llava_dpo_attention-loss)
      # Path to attention-loss DPO-tuned LLaVA model
      MODEL_PATH="/scratch/hs5580/mia/llava_lora_our_loss"
      ;;
    *)
      echo "Unknown model type: $MODEL_TYPE"
      exit 1
      ;;
  esac
fi

# Build command
CMD="python3 run_adaptvis_inference.py \
  --model-path \"$MODEL_PATH\" \
  --data-path \"$DATA_PATH\" \
  --image-dir \"$IMAGE_DIR\" \
  --output-path \"$OUTPUT_PATH\" \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --max-new-tokens $MAX_NEW_TOKENS \
  --precision $PRECISION \
  --image-aspect-ratio $IMAGE_ASPECT_RATIO"

# Add optional parameters
if [[ -n "$MODEL_BASE" ]]; then
  CMD="$CMD --model-base \"$MODEL_BASE\""
fi

if [[ "$USE_ADAPTVIS" = true ]]; then
  CMD="$CMD --use-adaptvis \
  --confidence-low-threshold $CONF_LOW \
  --confidence-high-threshold $CONF_HIGH \
  --alpha-low $ALPHA_LOW \
  --alpha-high $ALPHA_HIGH"
fi

# Add test mode parameters
if [[ "$TEST_MODE" = true ]]; then
  CMD="$CMD --test --test-size $TEST_SIZE --test-max-tokens $TEST_MAX_TOKENS"
  
  # If test mode is enabled, append _test to output path
  OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
  OUTPUT_FILE=$(basename "$OUTPUT_PATH")
  OUTPUT_NAME="${OUTPUT_FILE%.*}"
  OUTPUT_EXT="${OUTPUT_FILE##*.}"
  NEW_OUTPUT_PATH="$OUTPUT_DIR/${OUTPUT_NAME}_test.$OUTPUT_EXT"
  CMD="${CMD//$OUTPUT_PATH/$NEW_OUTPUT_PATH}"
  
  echo "TEST MODE ENABLED - Results will be saved to $NEW_OUTPUT_PATH"
fi

if [[ "$VERBOSE" = true ]]; then
  CMD="$CMD --verbose"
fi

# Print command
echo "Running command: $CMD"

# Execute command
eval $CMD 