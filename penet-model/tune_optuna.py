"""Optuna hyperparameter tuning for PENet / PENetClassifier.

Usage example (PowerShell):

    # Basic run optimizing validation AUROC over 20 trials, 3 epochs each
    python tune_optuna.py \
        --data_dir /users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/ \
        --dataset pe \
        --model PENetClassifier \
        --name optuna_pe \
        --num_slices 32 \
        --batch_size 6 \
        --n_trials 20 \
        --epochs_per_trial 3 \
        --metric val_AUROC

Notes:
    - Python version in current environment.yml is 3.6. Modern Optuna (>=3.x) requires >=3.8.
      If import fails, create a new environment with Python>=3.8 and install `optuna`.
    - The search space below is intentionally conservative to keep trials fast.
    - Metric direction inferred automatically (loss minimized, AUROC maximized).
    - Pruning can be enabled with --prune; it will prune after the first evaluation epoch.

Outputs:
    - Each trial uses a unique experiment name: <base_name>_trial<NUM>.
    - Individual trial args JSON and TensorBoard logs stored under each trial's generated save_dir.
    - Final study summary printed at end. Best params dumped to best_params.json in CWD.
"""

import json
import os
import sys
import argparse
import math
import torch
import torch.nn as nn

import util
import data_loader
import models

from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver

# Attempt to import optuna; provide helpful message if unavailable (e.g. Python 3.6 environment)
try:
    import optuna
    from optuna.trial import TrialState
except Exception as e:  # Broad to catch version / Python incompatibility
    print("[ERROR] Failed to import optuna. Ensure optuna is installed and Python >=3.8. Exception: {}".format(e))
    sys.exit(1)


def build_base_args(cli_args):
    """Parse base training args using existing TrainArgParser with provided list of args.

    Args:
        cli_args: List of argument tokens destined for TrainArgParser (excluding program name).
    """
    parser = TrainArgParser()
    args = parser.parse_args(cli_args)
    return args


def objective(trial, base_cli_args, study_cfg):
    """Optuna objective function.

    Args:
        trial: Optuna trial object.
        base_cli_args: List of CLI args destined for TrainArgParser (excluding optuna-only flags).
        study_cfg: Dict of study-level configuration (metric, epochs_per_trial, prune flag).
    Returns:
        Best metric value achieved during training for this trial.
    """
    args = build_base_args(base_cli_args)

    # Enforce constant hyperparameters requested by user
    if 'const' in study_cfg:
        args.num_slices = study_cfg['const']['num_slices']
        args.batch_size = study_cfg['const']['batch_size']

    # Overwrite experiment name for uniqueness per trial and rebuild save_dir
    import datetime
    args.name = f"{args.name}_trial{trial.number}"
    root_save_dir = os.path.dirname(args.save_dir)
    date_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_save_dir = os.path.join(root_save_dir, f'{args.name}_{date_string}')
    os.makedirs(new_save_dir, exist_ok=True)
    args.save_dir = new_save_dir
    # Persist updated args JSON under new directory
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=4, sort_keys=True)
        fh.write('\n')

    # Limit epochs for quicker trial turnaround
    args.num_epochs = study_cfg['epochs_per_trial']
    # Ensure evaluation happens each epoch
    args.epochs_per_eval = 1

    # Ensure logger divisibility constraints hold (iters_per_print must be multiple of batch_size)
    if args.iters_per_print % args.batch_size != 0:
        # Set to 1 * batch_size for brevity
        args.iters_per_print = args.batch_size
    if args.iters_per_visual % args.batch_size != 0:
        # Round up to nearest multiple of batch_size
        q = (args.iters_per_visual + args.batch_size - 1) // args.batch_size
        args.iters_per_visual = q * args.batch_size

    metric_key = study_cfg['metric']  # e.g. 'val_AUROC' or 'val_loss'
    # Set ckpt metric so ModelSaver tracks best (not strictly needed but consistent)
    args.best_ckpt_metric = 'val_loss' if metric_key == 'val_loss' else 'val_AUROC'
    args.maximize_metric = not args.best_ckpt_metric.endswith('loss')

    # ------------------------- Hyperparameter Search Space -------------------------
    # Optimizer & learning rates
    args.optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam'])
    args.learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    if args.optimizer == 'sgd':
        args.sgd_momentum = trial.suggest_float('sgd_momentum', 0.85, 0.99)
        args.sgd_dampening = trial.suggest_float('sgd_dampening', 0.0, 0.9)
    else:  # adam
        args.adam_beta_1 = trial.suggest_float('adam_beta_1', 0.85, 0.99)
        args.adam_beta_2 = trial.suggest_float('adam_beta_2', 0.90, 0.999)

    # Scheduler type fixed for now (could expose). Warmup steps scaled to epoch length heuristically.
    args.lr_scheduler = 'cosine_warmup'
    # Derive lr_decay_step (max iterations) as (#epochs * dataset_len/batch_size) after data loader built.

    # Regularization / architecture knobs
    args.dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5)

    # Pretrained / fine-tuning search space
    if args.use_pretrained:
        # Optionally toggle whether we fine-tune deeper layers
        args.fine_tune = trial.suggest_categorical('fine_tune', [True, False])
        if args.fine_tune:
            # LR for frozen part (earlier layers). Typically much smaller than head LR.
            args.fine_tuning_lr = trial.suggest_float('fine_tuning_lr', 1e-6, 1e-3, log=True)
            # Boundary after which layers are unfrozen (same semantics as existing code)
            args.fine_tuning_boundary = trial.suggest_categorical(
                'fine_tuning_boundary', ['encoders.0', 'encoders.1', 'encoders.2', 'encoders.3']
            )
        else:
            # Ensure no fine-tuning LR applied if not fine-tuning
            args.fine_tuning_lr = 0.0
    else:
        # Non-pretrained case: respect original flag
        if args.fine_tune:
            args.fine_tuning_lr = trial.suggest_float('fine_tuning_lr', 1e-5, 5e-3, log=True)

    # Data-related sampling probability
    args.abnormal_prob = trial.suggest_float('abnormal_prob', 0.3, 0.7)

    # --------------------------------------------------------------------------------

    # Build model & training components (largely mirroring train.py)
    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        if args.use_pretrained:
            model.load_pretrained(args.ckpt_path, args.gpu_ids)
        model = nn.DataParallel(model, args.gpu_ids)

    model = model.to(args.device)
    model.train()

    if args.use_pretrained or args.fine_tune:
        parameters = model.module.fine_tuning_parameters(args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)

    # Instantiate data loader to derive iteration counts for LR warmup & decay steps
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase='train', is_training=True)
    # Compute total iterations = epochs * ceil(dataset_len / batch_size)
    iters_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    args.lr_decay_step = iters_per_epoch * args.num_epochs
    lr_scheduler = util.get_scheduler(optimizer, args)

    cls_loss_fn = util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)
    logger = TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    eval_loaders = [data_loader_fn(args, phase='val', is_training=False)]
    evaluator = ModelEvaluator(args.do_classify, args.dataset, eval_loaders, logger,
                               args.agg_method, args.num_visuals, args.max_eval, args.epochs_per_eval)

    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    best_metric = None
    # Training loop with epoch-level evaluation & pruning support
    while not logger.is_finished_training():
        logger.start_epoch()
        for inputs, target_dict in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                inputs.to(args.device)
                cls_logits = model.forward(inputs)
                cls_targets = target_dict['is_abnormal']
                cls_loss = cls_loss_fn(cls_logits, cls_targets.to(args.device))
                loss = cls_loss.mean()
                logger.log_iter(inputs, cls_logits, target_dict, cls_loss.mean(), optimizer)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logger.end_iter()
            util.step_scheduler(lr_scheduler, global_step=logger.global_step)

        metrics, curves = evaluator.evaluate(model, args.device, logger.epoch)
        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
                   metric_val=metrics.get(args.best_ckpt_metric, None))
        logger.end_epoch(metrics, curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)

        current_metric = metrics.get(metric_key)
        if current_metric is not None:
            if best_metric is None:
                best_metric = current_metric
            else:
                if study_cfg['direction'] == 'maximize':
                    best_metric = max(best_metric, current_metric)
                else:
                    best_metric = min(best_metric, current_metric)

            # Report to Optuna & possibly prune
            trial.report(current_metric, step=logger.epoch)
            if study_cfg['prune'] and trial.should_prune():
                raise optuna.TrialPruned()

    # Fallback if metric missing
    if best_metric is None:
        best_metric = float('inf') if study_cfg['direction'] == 'minimize' else float('-inf')
    return best_metric


def parse_optuna_args():
    opt_parser = argparse.ArgumentParser(description='Optuna tuning wrapper')
    # Arguments forwarded to TrainArgParser should be passed normally; we intercept optuna-specific ones here.
    opt_parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials.')
    opt_parser.add_argument('--metric', type=str, default='val_AUROC', choices=('val_AUROC', 'val_loss'),
                            help='Validation metric to optimize.')
    opt_parser.add_argument('--epochs_per_trial', type=int, default=3, help='Epochs per trial.')
    opt_parser.add_argument('--study_name', type=str, default='optuna_study', help='Optuna study name.')
    opt_parser.add_argument('--storage', type=str, default='',
                            help='Optuna storage URI (e.g. sqlite:///optuna.db). If empty, use in-memory.')
    opt_parser.add_argument('--prune', type=util.str_to_bool, default=True, help='Enable pruning of bad trials.')
    # We allow passing arbitrary additional args for TrainArgParser after '--'
    known, unknown = opt_parser.parse_known_args()
    return known, unknown


def main():
    opt_args, train_cli_args = parse_optuna_args()

    # ------------------------------------------------------------------
    # Sanity adjustments to forwarded training args before trials begin.
    # Provide a default experiment name if missing.
    if '--name' not in train_cli_args:
        train_cli_args += ['--name', 'optuna_run']
    # If optimizing AUROC but user forgot classification flag, enable it.
    if opt_args.metric == 'val_AUROC' and '--do_classify' not in train_cli_args:
        train_cli_args += ['--do_classify', 'true']
    # ------------------------------------------------------------------

    direction = 'maximize' if not opt_args.metric.endswith('loss') else 'minimize'
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1) if opt_args.prune else optuna.pruners.NopPruner()
    study = optuna.create_study(direction=direction, study_name=opt_args.study_name,
                                storage=opt_args.storage if opt_args.storage else None,
                                load_if_exists=True, sampler=sampler, pruner=pruner)

    study_cfg = {
        'metric': opt_args.metric,
        'epochs_per_trial': opt_args.epochs_per_trial,
        'direction': direction,
        'prune': opt_args.prune
    }

    # Parse once to capture constant hyperparameters the user wants fixed
    preview_args = TrainArgParser().parse_args(train_cli_args)
    study_cfg['const'] = {
        'num_slices': preview_args.num_slices,
        'batch_size': preview_args.batch_size
    }

    def _objective(trial):
        return objective(trial, train_cli_args, study_cfg)

    print('[INFO] Starting Optuna study: {} (direction: {})'.format(opt_args.study_name, direction))
    study.optimize(_objective, n_trials=opt_args.n_trials, show_progress_bar=True)

    print('\n[RESULT] Study {} finished.'.format(opt_args.study_name))
    print('  Best trial number: {}'.format(study.best_trial.number))
    print('  Best value ({metric}): {value}'.format(metric=opt_args.metric, value=study.best_value))
    print('  Best params:')
    for k, v in study.best_params.items():
        print('    {}: {}'.format(k, v))

    # Persist best params to JSON for reuse
    with open('best_params.json', 'w') as fh:
        json.dump({'metric': opt_args.metric, 'value': study.best_value, 'params': study.best_params}, fh, indent=2)
        fh.write('\n')
    print('[INFO] Saved best params to best_params.json')


if __name__ == '__main__':
    util.set_spawn_enabled()
    main()

