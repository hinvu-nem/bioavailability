import numpy as np
import pandas as pd
import argparse
from helper.preprocess_dataset import split_dataset, convert_number, encode_categorical
from helper.features_selection import remove_noise_columns, remove_correlation
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, rand, hp, Trials, STATUS_OK
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Tune XGBoost hyperparameters with Hyperopt (TPE or Random search)')
parser.add_argument("--train", type=str, default='dataset/train_set.csv')
parser.add_argument("--target", type=str, default='bioavailability_percent')
parser.add_argument("--id_col", type=str, default='compound_id')
parser.add_argument("--categorical", type=str, default='formulation_type')
parser.add_argument("--n_trials", type=int, default=50, help="number of trials")
parser.add_argument("--cv_folds", type=int, default=5, help="number of fold")
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--algo", type=str, default='both', choices=['tpe', 'random', 'both'],help="Search algorithm: 'tpe', 'random', or 'both' (runs both and prints 2 tables)")

args = parser.parse_args()

print('* Loading dataset...')
train = pd.read_csv(args.train)
print(f'  | Train size: {len(train)}')
print('\n')

y_train = train[args.target]

categorical_cols = [c.strip() for c in args.categorical.split(',') if c.strip()]
drop_cols = [args.target, args.id_col] + categorical_cols
numeric_cols = [c for c in train.columns if c not in drop_cols]

X_numeric = train[numeric_cols].reset_index(drop=True)

if categorical_cols:
    X_cat = pd.get_dummies(train[categorical_cols], prefix=categorical_cols).astype(int)
    X_train_full = pd.concat([X_numeric, X_cat.reset_index(drop=True)], axis=1)
else:
    X_train_full = X_numeric

print(f'  | Feature size: {X_train_full.shape[1]}')
print('\n')

# Space for finding hyperparameter
space = {
    'n_estimators': hp.choice('n_estimators', [200, 300, 500, 800, 1000]),
    'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'min_child_weight': hp.choice('min_child_weight', [1, 2, 3, 4, 5]),
    'gamma': hp.uniform('gamma', 0, 0.5),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(10)),
}

n_estimators_choices = [200, 300, 500, 800, 1000]
max_depth_choices = [3, 4, 5, 6, 7, 8]
min_child_weight_choices = [1, 2, 3, 4, 5]

kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)


def objective(params):
    """
    Hàm mục tiêu: Hyperopt sẽ cố gắng tối thiểu giá trị trả về.
    """
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=args.random_state,
        n_jobs=-1,
    )

    scores = cross_val_score(model, X_train_full, y_train, cv=kf,
                              scoring='neg_root_mean_squared_error', n_jobs=1)
    rmse = -scores.mean()

    return {'loss': rmse, 'status': STATUS_OK}


def run_search(algo_name):
    # fmin for tpe or random
    search_algo = tpe.suggest if algo_name == 'tpe' else rand.suggest
    algo_label = 'TPE' if algo_name == 'tpe' else 'Random search'

    print(f'* Finding hyperparameters with {algo_label} ({args.n_trials} trials, {args.cv_folds}-fold CV)...')
    print('\n')

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=search_algo,
        max_evals=args.n_trials,
        trials=trials,
        rstate=np.random.default_rng(args.random_state),
    )

    best_params = {
        'n_estimators': n_estimators_choices[best['n_estimators']],
        'max_depth': max_depth_choices[best['max_depth']],
        'learning_rate': round(best['learning_rate'], 5),
        'subsample': round(best['subsample'], 4),
        'colsample_bytree': round(best['colsample_bytree'], 4),
        'min_child_weight': min_child_weight_choices[best['min_child_weight']],
        'gamma': round(best['gamma'], 4),
        'reg_alpha': round(best['reg_alpha'], 5),
        'reg_lambda': round(best['reg_lambda'], 5),
        'random_state': args.random_state,
        'n_jobs': -1,
    }

    best_rmse = min(trials.losses())

    print('\n')
    print(f'Best hyperparameters via {algo_label} (RMSE cross-validation = {best_rmse:.4f}):')
    print(tabulate([(k, v) for k, v in best_params.items()], headers=["Parameter", "Value"], tablefmt="grid"))
    print('\n')

    return algo_label, best_params, best_rmse


algos_to_run = ['tpe', 'random'] if args.algo == 'both' else [args.algo]
results = [run_search(a) for a in algos_to_run]

# Comparision
if len(results) > 1:
    print('=' * 60)
    print('Comparision results:')
    summary_rows = [(label, f'{rmse:.4f}') for label, _, rmse in results]
    print(tabulate(summary_rows, headers=["Algorithm", "Best RMSE (CV)"], tablefmt="grid"))
    print('\n')
