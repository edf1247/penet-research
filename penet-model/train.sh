#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:2
#SBATCH -t 48:00:00
#SBATCH -o /users/edfarber/train.log

/users/edfarber/penet/.venv/bin/python train.py \
  --data_dir=/users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0 \
  --ckpt_path=/users/edfarber/penet/penet_best.pth.tar \
  --save_dir=train_logs \
  --name=OptunaTest_regularized \
  --use_pretrained=True \
  --dataset=pe \
  --model=PENetClassifier \
  --do_classify=True \
  --agg_method=max \
  --best_ckpt_metric=val_AUROC \
  --gpu_ids=0,1 \
  --num_slices=24 \
  --batch_size=8 \
  --num_epochs=50 \
  --epochs_per_eval=1 \
  --epochs_per_save=5 \
  --optimizer=adam \
  --learning_rate=0.0001 \
  --weight_decay=1e-03 \
  --dropout_prob=0.5 \
  --fine_tune=True \
  --fine_tuning_lr=0 \
  --fine_tuning_boundary=encoders.3 \
  --abnormal_prob=0.5 \
  --lr_scheduler=plateau \
  --patience=10 \
  --lr_decay_gamma=0.5 \
  --crop_shape=192,192 \
  --resize_shape=208,208 \
  --include_normals=True \
  --do_hflip=True \
  --do_vflip=True \
  --do_rotate=True \
  --num_visuals=8 \
  --iters_per_print=8 \
  --iters_per_visual=8000 \
  --num_workers=8
