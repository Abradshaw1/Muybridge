#!/usr/bin/env bash
set -e
set -x

# Destination for models
data_dir=models
mkdir -p "$data_dir"
cd "$data_dir"

# Google Drive folder ID
FOLDER_ID="18vshM0p49UUyNOF0hxvAjbY7KmMcmuhU"

# Guard: skip if already downloaded
if [ -d "coreml_models" ]; then
    echo "Directory exists: coreml_models"
    exit 0
fi

# Download from Google Drive
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O coreml_models

echo "[MuyBridge] CoreML models downloaded into: $PWD/coreml_models"
