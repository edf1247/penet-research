#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/features.log

/users/edfarber/penet/.venv/bin/python extract_features.py --name xgboost_with_pooling --ckpt_path /users/edfarber/penet/penet_best.pth.tar --data_dir /users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/ --phase test --results_dir results/test_feats --dataset pe --gpu_ids 0
