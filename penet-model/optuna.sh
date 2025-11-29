#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -o /users/edfarber/optuna.log

/users/edfarber/penet/.venv/bin/python tune_optuna.py \
  --n_trials 50 \
  --epochs_per_trial 3 \
  --metric val_AUROC \
  --study_name penet_pe_optuna \
  --storage sqlite:////users/edfarber/optuna.db \
  --data_dir /users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/ \
  --use_pretrained true \
  --ckpt_path /users/edfarber/penet-research/penet-model/penet_best.pth.tar \
  --dataset pe \
  --model PENetClassifier \
  --name optuna_second_run \
  --gpu-ids 0,1 \
  --num_slices 24 \
  --batch_size 8 \
  --do_classify true \
  --agg_method max
