import os
import pickle
import numpy as np
import torch
import pandas as pd

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset.dataset_utils import get_data_in_tensor



def stratified_split(data_indices, data_labels, test_size=0.15, valid_size=0.10):
    # First split in training and testing
    X_train, X_test, y_train, y_test = train_test_split(data_indices, data_labels, stratify=data_labels, test_size=test_size,)

    # First split in training and testing
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size,)

    return {'X_train':X_train, 'X_valid':X_valid, 'X_test':X_test,
            'y_train':y_train, 'y_valid':y_valid, 'y_test':y_test}



def cv_stratified_split(data_indices, data_labels, k_folds=5, test_size=0.15):
    # First split in training and testing
    X_train, X_test, y_train, y_test = train_test_split(data_indices, data_labels, stratify=data_labels, test_size=test_size)

    # Now do cross-validation in training set
    cv_indices = list()
    skfold = StratifiedKFold(n_splits=k_folds, shuffle=False,)
    for idx, (train_index, test_index) in enumerate(skfold.split(X_train, y_train)):
        cv_indices.append(
            {'X_train': X_train[train_index], 'X_valid': X_train[test_index],
             'y_train': y_train[train_index], 'y_valid': y_train[test_index]})


    _, count_per_class_valid = np.unique(cv_indices[0]['y_valid'], return_counts=True)

    return cv_indices, {'X_test':X_test, 'y_test':y_test}



def cv_model(classifier,
             params,
             num_folds,
             cv_indices,
             test_idx,
             objective,
             rerun_best=False,
             in_params=None,
             ):
    list_params = []
    acc_res = np.zeros((num_folds, 1))
    y = objective.dataset.label.squeeze().data.numpy() 
    all_indices = np.arange(0, len(y))
    dataset = objective.dataset

    for ix_inner in range(0, num_folds):
        # print(f"Inner CV {ix_inner+1}/{num_folds}")

        train_idx = cv_indices[ix_inner]['X_train']
        valid_idx = cv_indices[ix_inner]['X_valid']
                
        # Explore the parameter space - it will run the different parameters configurations with the current CV split
        # max_resources -- needs to be set explicitly for XGBoost and RF
        # For the classifiers is : max_resources='n_samples' ; for XGBoost is max_resources='n_estimators'.
        if isinstance(classifier, xgb.XGBClassifier):
            max_resources = 20
            resource = 'max_depth'
        elif isinstance(classifier, RandomForestClassifier):
            max_resources = 20
            resource = 'max_depth'
        else:
            max_resources = 'auto'
            resource = 'n_samples'
        
        if rerun_best:
            split_params = in_params.query(f"Inner_Fold=={ix_inner}").copy()
            model_params = split_params[list(params.keys())].iloc[0].to_dict()
            list_params.append(model_params)
            if 'n_estimators' in model_params:
                model_params['n_estimators'] = int(model_params['n_estimators'])
            best_est = classifier.set_params(**model_params)
            search = None
        else:
            # We only care about the train indices for the normalisation --- this will run the normalization 
            objective.set_indices(train_idx, valid_idx, test_idx=test_idx, normal_group_idx=None, save_norm=False)

            # Get all the data of the dataset --- this will get the data using a dataloader and a collate to flattent it
            x, x_edges, label = get_data_in_tensor(dataset, all_indices, device='cpu')
            if objective.default_params['use_edges']:
                normX = torch.cat((x, x_edges), dim=1)
            else:
                normX = x

            search = HalvingGridSearchCV(classifier, param_grid=params, scoring='accuracy', cv=[(train_idx, valid_idx)], verbose=0, refit=False, 
                                        min_resources='exhaust', max_resources=max_resources, resource=resource)
            search.fit(normX.numpy(), y)
            list_params.append(search.best_params_)
            best_est = classifier.set_params(**search.best_params_)

        # ===== Normalize the data with the validation set and re-fit best estimator =====
        all_train = np.concatenate((train_idx, valid_idx))
        objective.set_indices(all_train, None, test_idx=test_idx, normal_group_idx=None, save_norm=False)
        x, x_edges, label = get_data_in_tensor(dataset, all_indices, device='cpu')
        if objective.default_params['use_edges']:
            normX = torch.cat((x, x_edges), dim=1)
        else:
            normX = x

        # Re-fit the best estimator        
        best_est.fit(normX[all_train].numpy(), y[all_train].copy())                        

        # Get the predictions on the test set
        pred_y = best_est.predict(normX[test_idx])
        acc_sp = accuracy_score(pred_y, y[test_idx].copy())
        acc_res[ix_inner] = acc_sp        

    return acc_res, list_params, search


def nested_cv_model(in_folds, 
                    out_folds, 
                    save_folder, 
                    params, 
                    classifier, 
                    test_size=0.2, 
                    load_previous=True, 
                    objective=None,
                    splits_file=None,
                    rerun_best=False,
                    ):
    
    if os.path.isfile(splits_file):        
        with open(splits_file, 'rb') as file:
            splits = pickle.load(file)
        out_folds = len(splits['cv_indices'])
        in_folds = len(splits['cv_indices'][0])
        predefined_splits = True
    else:
        predefined_splits = False

    # If available load the results
    model_path = os.path.join(save_folder, 'model.pkl')
    norm_info_path = os.path.join(save_folder, 'norm_info.pt')
    results_filename = os.path.join(save_folder, 'model_results.csv')
    params_filename = os.path.join(save_folder, 'params.csv')
    search_filename = os.path.join(save_folder, 'search.pkl')
    y = objective.dataset.label.squeeze().data.numpy() 

    if os.path.isfile(model_path) and load_previous:
        with open(model_path, 'rb') as file:
            best_est = pickle.load(file)
        
        norm_info = torch.load(norm_info_path)
        if rerun_best:
            acc_res = np.zeros((out_folds, in_folds))
            # Reload parameters
            df_params = pd.read_csv(os.path.join(save_folder, 'params.csv'), index_col=0)
            
            # Nested CV with the pre-found parameters
            for ix_out in range(0, out_folds):
                # Get the cross-validation indices
                if predefined_splits:
                    cv_indices = splits['cv_indices'][ix_out]
                    test_indices = splits['test_indices'][ix_out]
                else:
                    cv_indices, test_indices = cv_stratified_split(np.arange(0, len(y)), y, k_folds=in_folds, test_size=test_size)
                test_idx = test_indices['X_test']
                acc_cv, params_cv, search_cv = cv_model(classifier, 
                                            params, 
                                            in_folds, 
                                            cv_indices, 
                                            test_idx, 
                                            objective,                                         
                                            rerun_best=rerun_best,
                                            in_params=df_params.query(f"Outer_Fold=={ix_out}"),
                                            )
                acc_res[ix_out] = acc_cv.reshape(-1)
        else:            
            acc_res = pd.read_csv(results_filename, index_col=0).values
            df_params = pd.read_csv(os.path.join(save_folder, 'params.csv'), index_col=0)

        # Re-create the best estimator        
        best_params = df_params[list(params.keys())].iloc[np.argmax(acc_res.reshape(-1))]
        best_params = best_params.to_dict()
        if 'n_estimators' in best_params:
            best_params['n_estimators'] = int(best_params['n_estimators'])
        if 'max_depth' in best_params:
            best_params['max_depth'] = int(best_params['max_depth'])
        best_est = classifier.set_params(**best_params)

        # Indices for the train / valid split
        dataset = objective.dataset        
        train_idx = np.arange(0, len(y))
        # We only care about the train indices for the normalisation
        objective.set_indices(train_idx, None, None, normal_group_idx=None, save_norm=False)  

        # Get all the data of the dataset        
        x, x_edges, label = get_data_in_tensor(dataset, train_idx, device='cpu')
        if objective.default_params['use_edges']:
            normX = torch.cat((x, x_edges), dim=1)
        else:
            normX = x

        # Train 
        best_est.fit(normX.numpy(), y.copy())
        with open(model_path, 'wb') as file:
            pickle.dump(best_est, file)

        return best_est, acc_res, norm_info            
    else:
        acc_res = np.zeros((out_folds, in_folds))
        params_dict = dict()
        search_dict = dict()

        # Nested CV with parameter optimization
        for ix_out in range(0, out_folds):
            # Get the cross-validation indices
            if predefined_splits:
                cv_indices = splits['cv_indices'][ix_out]
                test_indices = splits['test_indices'][ix_out]
            else:
                cv_indices, test_indices = cv_stratified_split(np.arange(0, len(y)), y, k_folds=in_folds, test_size=test_size)
            test_idx = test_indices['X_test']
            acc_cv, params_cv, search_cv = cv_model(classifier, 
                                         params, 
                                         in_folds, 
                                         cv_indices, 
                                         test_idx, 
                                         objective,                                         
                                         )

            # Save the results
            search_dict[f"{ix_out}"] = search_cv
            params_dict[f"{ix_out}"] = params_cv
            # list_params = list_params + params_cv
            acc_res[ix_out] = acc_cv.reshape(-1)

        # Train a final one with the best parameters
        df_params = pd.concat([pd.DataFrame(params_dict[p]) for p in params_dict], keys=params_dict.keys())
        df_params.reset_index(inplace=True)
        df_params.rename(columns={'level_1':'Inner_Fold', 'level_0': 'Outer_Fold'}, inplace=True)

        best_params = df_params[list(params.keys())].iloc[np.argmax(acc_res.reshape(-1))]
        best_params = best_params.to_dict()
        if 'n_estimators' in best_params:
            best_params['n_estimators'] = int(best_params['n_estimators'])
        best_est = classifier.set_params(**best_params)

        # ===================== Train the model with all the data ===================
        # =================== NORMALIZE
        # Indices for the train / valid split
        dataset = objective.dataset        
        train_idx = np.arange(0, len(y))
        # We only care about the train indices for the normalisation
        objective.set_indices(train_idx, None, None, normal_group_idx=None, save_norm=False)  

        # Get all the data of the dataset        
        x, x_edges, label = get_data_in_tensor(dataset, train_idx, device='cpu')
        if objective.default_params['use_edges']:
            normX = torch.cat((x, x_edges), dim=1)
        else:
            normX = x

        # Save norm info
        norm_info = dataset._transform
        torch.save(norm_info, norm_info_path)                                

        # Store the results
        df_results = pd.DataFrame(data=acc_res, columns=[f'Fold_{x}' for x in range(0, in_folds)])
        df_results.to_csv(results_filename)

        # Store the parameters        
        df_params.to_csv(params_filename)

        # Store the search object
        with open(search_filename, 'wb') as file:
            pickle.dump(search_dict, file)

        # Train 
        best_est.fit(normX.numpy(), y.copy())
        with open(model_path, 'wb') as file:
            pickle.dump(best_est, file)

        return best_est, acc_res, norm_info





def cv_model_scaler(X, y,
                    classifier,
                    params,
                    num_folds,
                    cv_indices,
                    test_idx,
                    rerun_best=False,
                    in_params=None,
                    n_jobs=1,
                    use_scaler=False,):
    list_params = []
    acc_res = np.zeros((num_folds, 1))
        
    if isinstance(classifier, xgb.XGBClassifier):
        max_resources = 20
        resource = 'max_depth'
    elif isinstance(classifier, RandomForestClassifier):
        max_resources = 20
        resource = 'max_depth'
    else:
        max_resources = 'auto'
        resource = 'n_samples'
    
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    all_indices = np.arange(0, len(y))
    for ix_inner in range(num_folds):
        train_idx = cv_indices[ix_inner]['X_train']
        valid_idx = cv_indices[ix_inner]['X_valid']

        # Scale data
        if use_scaler:
            _ = scaler.fit_transform(X[train_idx])
            normX = scaler.transform(X.copy())
        else:    
            normX = X.copy()

        if rerun_best:
            split_params = in_params.query(f"Inner_Fold=={ix_inner}").copy()
            model_params = split_params[list(params.keys())].iloc[0].to_dict()
            list_params.append(model_params)
            if 'n_estimators' in model_params:
                model_params['n_estimators'] = int(model_params['n_estimators'])
            best_est = classifier.set_params(**model_params)
            search = None
        else:
            # search = HalvingGridSearchCV(classifier, param_grid=params, scoring='accuracy', cv=[(train_idx, valid_idx)], verbose=0, refit=False, 
            #                 min_resources='exhaust', max_resources=max_resources, resource=resource, n_jobs=n_jobs)
            search = GridSearchCV(classifier, param_grid=params, scoring='accuracy', cv=[(train_idx, valid_idx)], verbose=0, refit=False)
            search.fit(normX, y)
            list_params.append(search.best_params_)
            best_est = classifier.set_params(**search.best_params_)

        # Re-fit on combined training and validation data
        all_train = np.concatenate((train_idx, valid_idx))
        if use_scaler:
            _ = scaler.fit_transform(X[all_train])
            normX = scaler.transform(X.copy())
        best_est.fit(normX[all_train], y[all_train].copy())

        # Evaluate on the test set
        normX_test = normX[test_idx]
        pred_y = best_est.predict(normX_test)
        acc_sp = accuracy_score(pred_y, y[test_idx].copy())
        acc_res[ix_inner] = acc_sp

    return acc_res, list_params, search

