#!/bin/bash

# Default ONNX model path
ONNX_PATH=${1:-"onnx_export/model.onnx"}

# Extract directory and filename for output
ONNX_DIR=$(dirname "$ONNX_PATH")
ONNX_FILENAME=$(basename "$ONNX_PATH" .onnx)
ENGINE_PATH="${ONNX_DIR}/${ONNX_FILENAME}.engine"

# Check if ONNX file exists
if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX file '$ONNX_PATH' not found!"
    echo "Usage: $0 [path_to_onnx_file]"
    echo "Default path: onnx_export/model.onnx"
    exit 1
fi

echo "Converting ONNX model to TensorRT engine..."
echo "Input ONNX: $ONNX_PATH"
echo "Output Engine: $ENGINE_PATH"

# Convert ONNX to TensorRT with optimized settings
trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:4x3x224x224 \
    --maxShapes=input:8x3x224x224 \
    --workspace=4096 \
    --fp16 \
    --explicitBatch \
    --verbose

# Check conversion result
if [ $? -eq 0 ]; then
    echo "Successfully converted ONNX to TensorRT engine: $ENGINE_PATH"
    echo "Engine file size: $(du -h "$ENGINE_PATH" | cut -f1)"
else
    echo "Error: Failed to convert ONNX model to TensorRT engine"
    exit 1
fi
