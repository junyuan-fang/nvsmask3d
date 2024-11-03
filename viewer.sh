#!/bin/bash
#SBATCH --job-name=gsp_set_up
#SBATCH --account=project_2003267
#SBATCH --partition=gputest
#SBATCH --time=00:13:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100:1

ns-viewer nvsmask3d --load-config /scratch/project_2003267/nvsmask3d/outputs/5942004064_dslr_colmap/nvsmask3d/config.yml --viewer.make-share-url True