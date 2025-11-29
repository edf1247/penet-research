#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -o /users/edfarber/val.log

/users/edfarber/penet/.venv/bin/python validation.py --data_dir /users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0 \
                 --ckpt_path /users/edfarber/penet-research/penet-model/train_logs/OptunaTest_regularized_20251127_215720/best.pth.tar \
                 --results_dir results \
                 --phase val \
                 --name validation \
                 --dataset pe \
                 --gpu_ids 0,1