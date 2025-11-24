import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

import util
from args import TestArgParser
from data_loader import CTDataLoader
from saver import ModelSaver

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def extract_features(args):
    # Load model
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    model = model.to(args.device)
    model.eval()

    # Replace classifier with Identity to get features
    # Note: The dimension depends on the model architecture (likely 2048 for ResNet50-based)
    if hasattr(model.module, 'classifier') and hasattr(model.module.classifier, 'fc'):
        model.module.classifier.fc = Identity()
    else:
        print("Error: Model structure not as expected.")
        return

    data_loader = CTDataLoader(args, phase=args.phase, is_training=False)
    study2features = defaultdict(list)
    study2labels = {}

    print(f"Extracting features for phase: {args.phase}...")
    with torch.no_grad():
        for i, (inputs, targets_dict) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(args.device)
            output = model(inputs)
            
            # Handle tuple output if necessary
            features = output[0] if isinstance(output, tuple) else output
            features_np = features.cpu().numpy()

            for study_num, feat in zip(targets_dict['study_num'], features_np):
                study_num = int(study_num)
                study2features[study_num].append(feat)

                series = data_loader.get_series(study_num)
                if study_num not in study2labels:
                    study2labels[study_num] = int(series.is_positive)

    # --- STATISTICAL POOLING ---
    print("Pooling features (Max, Mean, Std)...")
    
    if not study2features:
        print("No features extracted.")
        return

    # Dynamically determine feature dimension from the first sample
    first_study_feats = next(iter(study2features.values()))
    feature_dim = first_study_feats[0].shape[0]
    print(f"Detected feature dimension: {feature_dim}")

    pooled_dim = feature_dim * 3 
    num_studies = len(study2features)
    
    xgb_inputs = np.zeros((num_studies, pooled_dim), dtype=np.float32)
    xgb_labels = np.zeros(num_studies, dtype=np.float32)
    
    sorted_study_nums = sorted(study2features.keys())
    
    for i, study_num in enumerate(sorted_study_nums):
        # shape: (num_windows, feature_dim)
        feats = np.array(study2features[study_num])
        
        # Compute statistics across the window dimension (axis 0)
        f_max = np.max(feats, axis=0)
        f_mean = np.mean(feats, axis=0)
        f_std = np.std(feats, axis=0)
        
        # Concatenate
        pooled = np.concatenate([f_max, f_mean, f_std])
        
        xgb_inputs[i] = pooled
        xgb_labels[i] = study2labels[study_num]

    # Save
    save_dir = os.path.join(args.results_dir, 'xgb_pooled')
    os.makedirs(save_dir, exist_ok=True)
    
    input_path = os.path.join(save_dir, f'{args.phase}_inputs.npy')
    label_path = os.path.join(save_dir, 'labels.npy')
    
    np.save(input_path, xgb_inputs)
    np.save(label_path, xgb_labels)
    print(f"Saved pooled features to {input_path}")

if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TestArgParser()
    args_ = parser.parse_args()
    extract_features(args_)
