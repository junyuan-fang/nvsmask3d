#!/bin/bash

# Define the base command
BASE_COMMAND="ns-train nvsmask3d --vis viewer --viewer.quit-on-train-completion True --experiment-name"

# Define the data directory
DATA_DIR="nvsmask3d/data/replica"

# Define the sequences
SEQUENCES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")

# Create the commands to run
commands=()
for sequence in "${SEQUENCES[@]}"; do
    commands+=("$BASE_COMMAND $sequence replica_nvsmask3d --data $DATA_DIR --sequence $sequence")
done

# Use parallel to run two jobs at a time
parallel --jobs 2 ::: "${commands[@]}"

echo "Training completed for all sequences."
