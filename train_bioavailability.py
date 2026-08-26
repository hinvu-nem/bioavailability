import numpy as np
import pandas as pd
import argparse
from helper.preprocess_dataset import split_dataset, convert_number, encode_categorical
from helper.features_selection import remove_noise_columns, remove_correlation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Train XBBoost Regression model to predict oral bioavailability')
parser.add_argument("--dataset", type=str, help="Path to unsplit dataset (csv format)")
parser.add_argument("--test_size", type=float, default=0.15, help="Size of test set (float from 0 to 1)")
parser.add_argument("--random_state", type=int, default=42, help="Set this to get reproducible results, default 42")
parser.add_argument("--train", type=str, default='dataset/train_set.csv', help="Path to train dataset (csv format)")
parser.add_argument("--test", type=str, default='dataset/test_set.csv', help="Path to test dataset (csv format)")
parser.add_argument("--target", type=str, default='bioavailability_percent')
parser.add_argument("--id_col", type=str, default='compound_id')
parser.add_argument("--categorical", type=str, default='formulation_type',
                     help="Comma-separated list of categorical columns to one-hot encode")
parser.add_argument("--zero_threshold", type=float, default=0.95,
                     help="Drop a feature if one value makes up more than this fraction of rows")
parser.add_argument("--correlation_threshold", type=float, default=0.90,
                     help="Drop one of any pair of features correlated above this threshold")
parser.add_argument("--model", type=str, default='xgb', help="xgb")
parser.add_argument("--save_model", type=str, default='xgb_bioavailability_model')
parser.add_argument("--save_plot", type=str, default='xgboost_results.png',
                     help="File ảnh lưu biểu đồ feature importance + predicted vs actual")
parser.add_argument("--top_n_features", type=int, default=15,
                     help="Số lượng đặc trưng hiển thị trên biểu đồ importance")

args = parser.parse_args()

print('* Loading dataset...')

# Define dataset
if args.dataset is None:
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
else:
    train, test = split_dataset(pd.read_csv(args.dataset), test_size=args.test_size, random_state=args.random_state)

print(f'  | Train size: {len(train)}')
print(f'  | Test size: {len(test)}')
print('\n')

# Define target variables

y_train = train[args.target]
y_test = test[args.target]

# one-hot encoded 
print('* one-hot encoded ')
print('\n')

categorical_cols = [c.strip() for c in args.categorical.split(',') if c.strip()]
drop_cols = [args.target, args.id_col] + categorical_cols

numeric_cols = [c for c in train.columns if c not in drop_cols]

X_train_numeric = train[numeric_cols].reset_index(drop=True)
X_test_numeric = test[numeric_cols].reset_index(drop=True)

if categorical_cols:
    X_train_cat, X_test_cat = encode_categorical(train, test, categorical_cols)
    X_train_full = pd.concat([X_train_numeric, X_train_cat.reset_index(drop=True)], axis=1)
    X_test_full = pd.concat([X_test_numeric, X_test_cat.reset_index(drop=True)], axis=1)
else:
    X_train_full = X_train_numeric
    X_test_full = X_test_numeric

# Apply feature selection

print('* Apply features selection...')

X_train_full_fil = remove_noise_columns(X_train_full, threshold=args.zero_threshold)
X_train_full_fil = remove_correlation(X_train_full_fil, threshold=args.correlation_threshold)

print('\n')
print(f'  | Feature size after selection: {X_train_full_fil.shape[1]}')
print(f'  | Features used: {list(X_train_full_fil.columns)}')
print('\n')

# Train model

if args.model == 'xgb':
    with open('ml_config/xgboost_config.txt', 'r') as f:
        params = {k: convert_number(v) for k, v in (line.strip().split('=') for line in f if '=' in line)}

    table_data = [(k, v) for k, v in params.items()]

    model = XGBRegressor(**params)
    model_name = 'Extreme Gradient Boosting'
else:
    raise ValueError('This model does not provide other model')
print(f'Start training model {model_name} with following parameters:')

print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

train_model = model.fit(X_train_full_fil, y_train)
print('\n')
print(f'Train model {model_name} successfully. Calculating performance...')
print('\n')

X_test_full_fil = X_test_full[X_train_full_fil.columns]

y_train_hat = train_model.predict(X_train_full_fil)
y_test_hat = train_model.predict(X_test_full_fil)

r2_train = r2_score(y_train, y_train_hat)
r2_test = r2_score(y_test, y_test_hat)

mse_train = mean_squared_error(y_train, y_train_hat)
mse_test = mean_squared_error(y_test, y_test_hat)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

mae_train = mean_absolute_error(y_train, y_train_hat)
mae_test = mean_absolute_error(y_test, y_test_hat)

print(f'Model {model_name} metrics:')

headers = ["Metrics", "Train", "Test"]

data = [
    ["R2", round(r2_train, 4), round(r2_test, 4)],
    ["MSE", round(mse_train, 2), round(mse_test, 2)],
    ["RMSE", round(rmse_train, 2), round(rmse_test, 2)],
    ["MAE", round(mae_train, 2), round(mae_test, 2)]
]

print(tabulate(data, headers=headers, tablefmt="grid"))
print('\n')

# Feature importance (XGB)
if args.model in ('xgb', 'rf'):
    importances = pd.Series(train_model.feature_importances_, index=X_train_full_fil.columns)
    importances = importances.sort_values(ascending=False)
    print('Top 10 feature importances:')
    print(tabulate(importances.head(10).reset_index().values.tolist(),
                    headers=["Feature", "Importance"], tablefmt="grid"))
    print('\n')

#Vẽ biểu đồ feature importance + predicted vs actual
print('* Vẽ biểu đồ kết quả...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
#Feature importance plot
top_importance = importances.head(args.top_n_features).sort_values(ascending=True)
top_importance.plot(kind="barh", ax=axes[0], color="#2E86AB")
axes[0].invert_yaxis()
axes[0].set_title(f"Tầm quan trọng đặc trưng ({model_name})")
axes[0].set_xlabel("Importance (gain-based, mặc định weight)")
 
#5b. Predicted vs Actual (trên tập test)
axes[1].scatter(y_test, y_test_hat, alpha=0.4, color="#A23B72", s=18)
lims = [min(y_test.min(), y_test_hat.min()), max(y_test.max(), y_test_hat.max())]
axes[1].plot(lims, lims, "k--", lw=1.5, label="Dự đoán hoàn hảo")
axes[1].set_xlabel("Bioavailability thực tế (%)")
axes[1].set_ylabel("Bioavailability dự đoán (%)")
axes[1].set_title(f"Dự đoán vs Thực tế (R²={r2_test:.3f})")
axes[1].legend()
 
plt.tight_layout()
plt.savefig(args.save_plot, dpi=150)
plt.close(fig)
print(f"Đã lưu biểu đồ: {args.save_plot}")
print('\n')


print(f'Saving model to file name {args.save_model}.pkl')

with open(f'{args.save_model}.pkl', 'wb') as model_file:
    pickle.dump({'model': model, 'feature_columns': list(X_train_full_fil.columns)}, model_file)

print(f'Save model successfully to file {args.save_model}.pkl')
