#!/bin/bash
#SBATCH --job-name=scannetpp_eval
#SBATCH --account=project_2003267
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:v100:1,nvme:30

###############ns-eval --load-config /scratch/project_2003267/nvsmask3d/outputs/5942004064_dslr_colmap/nvsmask3d/config.yml --render-output-path /scratch/project_2003267/nvsmask3d/eval/5942004064
echo "bash is running"
python3 /scratch/project_2003267/nvsmask3d/nvsmask3d/automation_test.py # logics is on 