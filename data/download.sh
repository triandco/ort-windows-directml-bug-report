#!/bin/bash

BASE_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/directml/directml-int4-awq-block-128/"
FILES=(
    "model.onnx"
    "model.onnx.data"
    "tokenizer.json"
)

for FILE in "${FILES[@]}"; do
    wget "${BASE_URL}${FILE}"
done
