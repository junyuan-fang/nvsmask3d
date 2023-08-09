#!/bin/bash
#SBATCH --job-name=gsp_set_up
#SBATCH --account=project_2003267
#SBATCH --partition=gpusmall
#SBATCH --time=30:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1,nvme:2000
#SBATCH --output=/scratch/project_2003267/junyuan/nvsmask3d/cluster/mahti/%x-%j.out  # 保存默认文件名到 /mahti
##SBATCH --error=/scratch/project_2003267/junyuan/nvsmask3d/cluster/mahti/%x-%j.err  # 保存默认文件名到 /mahti

scenes=(
    "7b6477cb95" "c50d2d1d42" "cc5237fd77" "acd95847c5" "fb5a96b1a2"
    "a24f64f7fb" "1ada7a0617" "5eb31827b7" "3e8bba0176" "3f15a9266d"
    "21d970d8de" "5748ce6f01" "c4c04e6d6c" "7831862f02" "bde1e479ad"
    "38d58a7a31" "5ee7c22ba0" "f9f95681fd" "3864514494" "40aec5fffa"
    "13c3e046d7" "e398684d27" "a8bf42d646" "45b0dac5e3" "31a2c91c43"
    "e7af285f7d" "286b55a2bf" "7bc286c1b6" "f3685d06a9" "b0a08200c9"
    "825d228aec" "a980334473" "f2dc06b1d2" "5942004064" "25f3b7a318"
    "bcd2436daf" "f3d64c30f8" "0d2ee665be" "3db0a1c8f3" "ac48a9b736"
    "c5439f4607" "578511c8a9" "d755b3d9d8" "99fa5c25e1" "09c1414f1b"
    "5f99900f09" "9071e139d9" "6115eddb86" "27dd4da69e" "c49a8c6cff"
)
# scenes=(
#     '5748ce6f01' '9071e139d9' '578511c8a9' 'c49a8c6cff' '5f99900f09' '1ada7a0617' '09c1414f1b' '27dd4da69e' '6115eddb86'
# )
# scenes=(
#     '09c1414f1b'
# )

output_base_dir="/scratch/project_2003267/junyuan/nvsmask3d/outputs"

for scene in "${scenes[@]}"; do
    echo "Starting training for scene $scene"

    ns-train nvsmask3d \
        --experiment-name "${scene}_dslr_colmap" \
        --timestamp "" \
        --vis viewer scannetpp_nvsmask3d \
        --data /scratch/project_2003267/junyuan/nvsmask3d/nvsmask3d/data/ScannetPP \
        --sequence "$scene" \
        --mode dslr_colmap &
    train_pid=$!

    echo "$train_pid"

    checkpoint_file="${output_base_dir}/${scene}_dslr_colmap/nvsmask3d/nerfstudio_models/step-000019999.ckpt"

    echo "Monitoring for completion of training for scene $scene..."
    while [ ! -f "$checkpoint_file" ]; do
        sleep 20
    done

    echo "Training checkpoint detected for scene $scene. Terminating training process."
    kill -9 "$train_pid"

    wait "$train_pid"

    echo "Training for scene $scene completed. Proceeding to the next scene."
done

echo "All scenes have been processed."
