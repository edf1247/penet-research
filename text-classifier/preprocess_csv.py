import pandas as pd
import os

data_path = r"PATH TO YOUR DATA"

labels = pd.read_csv(os.path.join(data_path, "Labels.csv"))
demographics = pd.read_csv(os.path.join(data_path, "Demographics.csv"))
icd = pd.read_csv(os.path.join(data_path, "ICD.csv"))
inp_med = pd.read_csv(os.path.join(data_path, "INP_MED.csv"))
out_med = pd.read_csv(os.path.join(data_path, "OUT_MED.csv"))
labs = pd.read_csv(os.path.join(data_path, "LABS.csv"))
vitals = pd.read_csv(os.path.join(data_path, "Vitals.csv"))

def clean_df(df, keep_split=False):
    cols_to_drop = [col for col in df.columns if col.startswith('Unnamed')]
    if not keep_split:
        cols_to_drop.append('split')
    df = df.drop(columns=cols_to_drop, errors='ignore')
    # Remove duplicate idx values (keep first occurrence)
    if df['idx'].duplicated().any():
        print(f"  Warning: Dropping {df['idx'].duplicated().sum()} duplicate idx rows")
        df = df.drop_duplicates(subset='idx', keep='first')
    return df

labels = clean_df(labels, keep_split=True)
demographics = clean_df(demographics)
icd = clean_df(icd)
inp_med = clean_df(inp_med)
out_med = clean_df(out_med)
labs = clean_df(labs)
vitals = clean_df(vitals)

merged = labels.copy()
merged = merged.merge(demographics, on='idx', how='inner')
merged = merged.merge(icd, on='idx', how='inner')
merged = merged.merge(inp_med, on='idx', how='inner')
merged = merged.merge(out_med, on='idx', how='inner')
merged = merged.merge(labs, on='idx', how='inner')
merged = merged.merge(vitals, on='idx', how='inner')

output_path = os.path.join(data_path, "merged_pe_data.csv")
merged.to_csv(output_path, index=False)
