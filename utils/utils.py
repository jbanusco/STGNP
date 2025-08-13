import os
import random
import torch
import numpy as np
import argparse
import scipy as sp

# Try nested CV one day...
# https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-rightdee
# https://stats.stackexchange.com/questions/11602/training-on-the-full-dataset-after-cross-validation
#     # Nested CV with parameter optimization
#     clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
#     nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
#     nested_scores[i] = nested_score.mean()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed=5593):
    random.seed(seed)
    max_int = int(1E6)
    tseed = random.randint(1, max_int)
    tcseed = random.randint(1, max_int)
    npseed =random.randint(1, max_int)
    ospyseed = random.randint(1, max_int)
    torch.manual_seed(tseed)
    torch.cuda.manual_seed(tcseed)
    np.random.seed(npseed)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)
    # torch.backends.cudnn.deterministic = True


def get_best_params(df_params, param_prefix='params_', use_median=False, binary_threshold=2):
    """
    Extracts the best parameters from df_params:
    - Uses mode for binary/categorical parameters (<= binary_threshold unique values)
    - Uses mean or median for continuous parameters

    Args:
        df_params: pandas DataFrame
        param_prefix: prefix for parameter columns (e.g. 'params_')
        use_median: if True, use median instead of mean for continuous params
        binary_threshold: max number of unique values to treat as categorical

    Returns:
        dict of parameter names (without prefix) and selected values
    """
    param_cols = [col for col in df_params.columns if col.startswith(param_prefix)]
    best_params = {}

    for col in param_cols:
        values = df_params[col].dropna()
        unique_vals = values.unique()

        if df_params[col].dtype == 'bool':
            # Treat as binary/categorical
            most_common = bool(sp.stats.mode(values.values.astype(float))[0])
            best_params[col.replace(param_prefix, '')] = most_common
        else:
            # Treat as continuous
            agg_value = values.median() if use_median else values.mean()
            best_params[col.replace(param_prefix, '')] = agg_value
            if df_params[col].dtype == 'int64':
                # If it's an integer column, convert to int
                best_params[col.replace(param_prefix, '')] = int(agg_value)
            

    return best_params
