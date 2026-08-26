import numpy as np
import pandas as pd
import argparse
import pickle
from tabulate import tabulate
import datetime
import os

parser = argparse.ArgumentParser(description='Predict bioavailability_percent for new compounds')
parser.add_argument("--file", type=str,default= 'dataset/new_compounds_to_predict.csv', help="CSV file of compounds to predict")
parser.add_argument("--id_col", type=str, default='compound_id')
parser.add_argument("--categorical", type=str, default='formulation_type',help="Comma-separated categorical columns that were one-hot encoded at train time")
parser.add_argument("--model", type=str, default='xgb_bioavailability_model.pkl')
parser.add_argument("--save_dir", type=str, default='predict_result')

# Single-compound mode (used when --file is not provided)
parser.add_argument("--molecular_weight", type=float)
parser.add_argument("--logP", type=float)
parser.add_argument("--tpsa", type=float)
parser.add_argument("--hbd", type=int)
parser.add_argument("--hba", type=int)
parser.add_argument("--rotatable_bonds", type=int)
parser.add_argument("--aromatic_rings", type=int)
parser.add_argument("--logS", type=float)
parser.add_argument("--dose_mg", type=float)
parser.add_argument("--particle_size_um", type=float)
parser.add_argument("--ph_stability", type=float)
parser.add_argument("--formulation_type", type=str, choices=['tablet', 'capsule', 'solution', 'suspension'])

args = parser.parse_args()

# Loading model 
with open(args.model, 'rb') as f:
    saved = pickle.load(f)

xgb_model = saved['model']
feature_cols = saved['feature_columns']
categorical_cols = [c.strip() for c in args.categorical.split(',') if c.strip()]

if args.file:
    pred = pd.read_csv(args.file)
    ids = pred[args.id_col] if args.id_col in pred.columns else pd.Series(range(len(pred)), name='row')

elif args.molecular_weight is not None:
    single_row = {
        'molecular_weight': args.molecular_weight,
        'logP': args.logP,
        'tpsa': args.tpsa,
        'hbd': args.hbd,
        'hba': args.hba,
        'rotatable_bonds': args.rotatable_bonds,
        'aromatic_rings': args.aromatic_rings,
        'logS': args.logS,
        'dose_mg': args.dose_mg,
        'particle_size_um': args.particle_size_um,
        'ph_stability': args.ph_stability,
        'formulation_type': args.formulation_type,
    }
    pred = pd.DataFrame([single_row])
    ids = pd.Series(['compound_1'], name=args.id_col)

else:
    raise ValueError('Provide --file (CSV of compounds) or the individual descriptor flags for a single compound')

# Rebuild the same feature matrix shape as training: numeric cols + one-hot categorical cols,
# Reindexed to the exact columns/order the model was trained on (missing dummy cols -> 0)
numeric_cols = [c for c in pred.columns if c not in categorical_cols + [args.id_col]]
X_numeric = pred[numeric_cols].reset_index(drop=True)

if categorical_cols:
    X_cat = pd.get_dummies(pred[categorical_cols], prefix=categorical_cols).astype(int)
    X_full = pd.concat([X_numeric, X_cat.reset_index(drop=True)], axis=1)
else:
    X_full = X_numeric

# Align to training feature columns exactly (adds any missing dummy cols as 0, drops extras)
X_predict = X_full.reindex(columns=feature_cols, fill_value=0)

pred_val = xgb_model.predict(X_predict)

df = pd.DataFrame({
    args.id_col: ids.values,
    "pred_bioavailability_percent": np.round(pred_val, 2)
})

# Print as a table
print(tabulate(df, headers="keys", tablefmt="grid"))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = args.save_dir if args.save_dir else "predict_result"
os.makedirs(save_dir, exist_ok=True)

# Save to CSV
save_path = f'{save_dir}_prediction.csv'
df.to_csv(save_path, index=False)

print(f"Saved prediction to file {save_path}")
