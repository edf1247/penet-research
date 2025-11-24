#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/xgboost_train.log

/users/edfarber/penet/.venv/bin/python run_xgboost.py \
    --train_path results/train_feats/xgboost_with_pooling_20251123_014542/xgb_pooled/train_inputs.npy \
    --val_path results/val_feats/xgboost_with_pooling_20251123_015912/xgb_pooled/val_inputs.npy \
    --name xgb_with_pooling \
    --model_dir results/xgb_models \
    --overwrite true \
    --num_iters 50 \
    --max_depth 4 \
    --learning_rate 0.1
