#!/bin/bash

# Define project name
PROJECT_NAME="group_activity_recognition"

# Define folder structure
FOLDERS=(
    "$PROJECT_NAME/data/raw"
    "$PROJECT_NAME/data/processed"
    "$PROJECT_NAME/models"
    "$PROJECT_NAME/training"
    "$PROJECT_NAME/evaluation"
    "$PROJECT_NAME/utils"
    "$PROJECT_NAME/experiments/exp1"
    "$PROJECT_NAME/experiments/exp2"
    "$PROJECT_NAME/notebooks"
)

# Define files and their initial content
declare -A FILES
FILES["$PROJECT_NAME/data/data_loader.py"]="# Script to load and preprocess dataset"
FILES["$PROJECT_NAME/data/dataset_utils.py"]="# Utilities for dataset handling"
FILES["$PROJECT_NAME/models/individual_lstm.py"]="# Individual-level LSTM model"
FILES["$PROJECT_NAME/models/group_lstm.py"]="# Group-level LSTM model"
FILES["$PROJECT_NAME/models/hierarchical_model.py"]="# Full hierarchical model"
FILES["$PROJECT_NAME/training/train.py"]="# Main training script"
FILES["$PROJECT_NAME/training/trainer.py"]="# Training utilities"
FILES["$PROJECT_NAME/training/config.py"]="# Configuration parameters"
FILES["$PROJECT_NAME/evaluation/evaluate.py"]="# Model evaluation script"
FILES["$PROJECT_NAME/evaluation/metrics.py"]="# Performance metrics (accuracy, F1-score, etc.)"
FILES["$PROJECT_NAME/evaluation/visualize.py"]="# Visualization utilities"
FILES["$PROJECT_NAME/utils/logging.py"]="# Logging utilities"
FILES["$PROJECT_NAME/utils/checkpoint.py"]="# Checkpoint saving/loading"
FILES["$PROJECT_NAME/utils/plot.py"]="# Plotting utilities"
FILES["$PROJECT_NAME/requirements.txt"]="# Add dependencies here"
FILES["$PROJECT_NAME/README.md"]="# Project Documentation"
FILES["$PROJECT_NAME/.gitignore"]="# Ignore unnecessary files"

# Create folders
for folder in "${FOLDERS[@]}"; do
    mkdir -p "$folder"
done

# Create files with initial content
for file in "${!FILES[@]}"; do
    echo -e "${FILES[$file]}" > "$file"
done

echo "Project '$PROJECT_NAME' structure created successfully! 🚀"
