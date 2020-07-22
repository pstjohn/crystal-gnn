#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=crystal_gnn
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/gpu.%j.out

source ~/.bashrc
module load cudnn/7.4.2/cuda-10.0
conda activate /projects/rlmolecule/pstjohn/envs/tf2_pymatgen

srun python train_model.py
