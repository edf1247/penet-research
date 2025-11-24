#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/test.log

/users/edfarber/penet/.venv/bin/python test.py --data_dir /users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0 \
                 --ckpt_path optuna_smoke_trial4_20251124_070317 \
                 --results_dir results \
                 --phase test \
                 --name test_with_optuna \
                 --dataset pe \
                 --gpu_ids 0
