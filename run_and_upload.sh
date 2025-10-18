#!/bin/bash

# Exit on error
set -e

# Default values
DTYPE="bf16"
GPU_INDEX="0"
HF_DATASET="muellerzr/consumer-mamf"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --gpu-index)
            GPU_INDEX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dtype DTYPE] [--gpu-index INDEX]"
            echo "  --dtype: Data type used (default: bf16)"
            echo "  --gpu-index: GPU index to use (default: 1)"
            exit 1
            ;;
    esac
done

# Generate timestamp for output file
TIMESTAMP=$(date +'%Y-%m-%d-%H:%M:%S')
OUTPUT_FILE="${TIMESTAMP}.txt"

# Detect GPU model from the specified index
echo "Detecting GPU model from index $GPU_INDEX..."
GPU_MODEL=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i "$GPU_INDEX" 2>/dev/null | sed 's/^[ \t]*//;s/[ \t]*$//' || echo "unknown_gpu")

# Run the MAMF finder
echo "Running MAMF finder on GPU: $GPU_MODEL..."
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python mamf-finder.py \
    --m_range 0 16384 1024 \
    --n_range 0 16384 1024 \
    --k_range 0 16384 1024 \
    --dtype "${DTYPE/bf16/bfloat16}" \
    --output_file="$OUTPUT_FILE"

# Clean up GPU model name for filename
GPU_MODEL_CLEAN=$(echo "$GPU_MODEL" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -d '/')

# Create filename for HuggingFace
HF_FILENAME="${GPU_MODEL_CLEAN}_${DTYPE}.txt"

echo "GPU Model: $GPU_MODEL"
echo "Dtype: $DTYPE"
echo "Output file: $OUTPUT_FILE"
echo "HuggingFace filename: $HF_FILENAME"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Please install with: pip install huggingface-hub[cli]"
    exit 1
fi

# Upload to HuggingFace
echo "Uploading to HuggingFace dataset: $HF_DATASET..."
huggingface-cli upload "$HF_DATASET" "$OUTPUT_FILE" "$HF_FILENAME" --repo-type dataset

echo "Upload complete!"
echo "File uploaded as: $HF_FILENAME to $HF_DATASET"
