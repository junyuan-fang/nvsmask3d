#!/bin/bash
#SBATCH --job-name=scannetpp_check
#SBATCH --account=project_2003267
#SBATCH --partition=gputest
#SBATCH --time=00:13:80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:v100:4,nvme:30

###############ns-eval --load-config /scratch/project_2003267/nvsmask3d/outputs/5942004064_dslr_colmap/nvsmask3d/config.yml --render-output-path /scratch/project_2003267/nvsmask3d/eval/5942004064
echo "bash is running"
python3 /scratch/project_2003267/nvsmask3d/check_depthmap.py