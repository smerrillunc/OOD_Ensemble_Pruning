import pandas as pd
import numpy as np

import sys, os
import random
import numbers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.integrate import trapezoid
from joblib import load, dump
from time import sleep

def get_dataset(dataset_path, dataset_name):
    """
    Description: Retrieve a chosen dataset
    """

    table_shift = ["diabetes_readmission",
                    "anes",
                    "assistments",
                    "nhanes_lead",
                    "college_scorecard",
                    "brfss_diabetes",
                    "acsfoodstamps",
                    "heloc",
                    "brfss_blood_pressure",
                    "mimic_extract_los_3",
                    "mimic_extract_mort_hosp",
                    "acsincome",
                    "acspubcov",
                    "physionet",
                    "acsunemployment",
                    'compas',
                    'german']

    if dataset_name == "CHEM":
        with open(f'{dataset_path}/CHEMOOD/train.csv', 'r') as f:
            X = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])
        with open(f'{dataset_path}/CHEMOOD/train.csv', 'r') as f:
            Y = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_id.csv', 'r') as f:
            X_val = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_id.csv', 'r') as f:
            Y_val = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_ood.csv', 'r') as f:
            X_val_ood = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_ood.csv', 'r') as f:
            Y_val_ood = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

        # with open(f'{dataset_path}/CHEMOOD/test_id.csv', 'r') as f:
        #    X_test = torch.from_numpy(np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:]))

        # with open(f'{dataset_path}/CHEMOOD/test_id.csv', 'r') as f:
        #    Y_test = torch.from_numpy(np.float32(np.array([line.strip().split(',')[1] for line in f])[1:]))

        #with open(f'{dataset_path}/CHEMOOD/test_ood.csv', 'r') as f:
        #    X_test_ood = torch.from_numpy(np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:]))

        #with open(f'{dataset_path}/CHEMOOD/test_ood.csv', 'r') as f:
        #    Y_test_ood = torch.from_numpy(np.float32(np.array([line.strip().split(',')[1] for line in f])[1:]))


    elif dataset_name in ['adult_tf', 'heloc_tf', 'yeast_tf', 'synthetic', 'hosptial_tf']:
        X = np.load(f'{dataset_path}/{dataset_name}/xs_train.npy')
        Y = np.load(f'{dataset_path}/{dataset_name}/ys_train.npy')

        X_val = np.load(f'{dataset_path}/{dataset_name}/xs_val_id.npy')
        Y_val = np.load(f'{dataset_path}/{dataset_name}/ys_val_id.npy')

        X_val_ood = np.load(f'{dataset_path}/{dataset_name}/xs_val_ood.npy')
        Y_val_ood = np.load(f'{dataset_path}/{dataset_name}/ys_val_ood.npy')

    elif dataset_name in table_shift:
        X = np.loadtxt(f'{dataset_path}/{dataset_name}/xs_train.csv', delimiter=',', skiprows=1)
        Y = np.loadtxt(f'{dataset_path}/{dataset_name}/ys_train.csv', delimiter=',', skiprows=1)

        X_val = np.loadtxt(f'{dataset_path}/{dataset_name}/xs_val_id.csv', delimiter=',', skiprows=1)
        Y_val = np.loadtxt(f'{dataset_path}/{dataset_name}/ys_val_id.csv', delimiter=',', skiprows=1)

        X_val_ood = np.loadtxt(f'{dataset_path}/{dataset_name}/xs_val_ood.csv', delimiter=',', skiprows=1)
        Y_val_ood = np.loadtxt(f'{dataset_path}/{dataset_name}/ys_val_ood.csv', delimiter=',', skiprows=1)

    else:
        print("Please specify a valid dataset and re-run the script")
        return 0

    return X, Y.ravel(), X_val, Y_val.ravel(), X_val_ood, Y_val_ood.ravel()

def create_directory_if_not_exists(directory):
    """
    Description: Create a new file if one doesn't exist
    """

    i = 0
    while True:
        try:
            if not os.path.exists(directory + f'/{i}'):
                os.makedirs(directory + f'/{i}')
                print(f"Directory '{directory + f'/{i}'}' created.")
                return directory + f'/{i}'
            else:
                print(f"Directory '{directory + f'/{i}'}' already exists.")
                i += 1

        except Exception as e:
            sleep(3)


def save_dict_to_file(dictionary, filename):
    """
    Description: Save a dictionary to a file
    """

    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def load_model(filename):
    loaded_clf = load(filename)
    return loaded_clf



def train_model(estimator, x_train, y_train, training_frac, save_path):
    num_features = x_train.shape[1]
    filename = f"{save_path}/models/{estimator.random_state}_{estimator.feature_ones}_{str(training_frac)}.json"

    features_encoding = generate_binary_vector(num_features, estimator.feature_ones, seed=estimator.random_state)

    sampled_indices, unsampled_indices = generate_sample_indices(estimator.random_state, 
                                                                 x_train.shape[0],
                                                                 training_frac)
    
    estimator.fit(x_train[sampled_indices][:,np.where(features_encoding==1)[0]], y_train[sampled_indices])
    
    # save estimator
    dump(estimator, filename, compress=9)

    return estimator, filename


def auprc_threshs(confidences, Y, threshs, as_percentages=True):
    areas = []
    sorti = np.argsort(-confidences)
    sortY = Y[sorti]
    precisions = np.cumsum(sortY)/np.arange(1, len(sortY)+1)
    recalls = np.cumsum(sortY)/np.sum(sortY)
    for t in threshs:
        interp_prec_t = np.interp(t, recalls, precisions)
        recalls_thresh = np.concatenate(([0.0], recalls[recalls<t], [t]))
        precisions_thresh = np.concatenate(([1.0], precisions[recalls<t], [interp_prec_t]))
        area_t = trapezoid(precisions_thresh, recalls_thresh)
        # area_t = metrics.auc(recalls_thresh, precisions_thresh)
        if as_percentages:
            area_t = area_t/t
        areas.append(area_t)
        
    return np.array(areas)

def get_precision_recall_auc(ensemble_pred_probs, y_val_ood, AUCTHRESHS):
    ensemble_preds = ensemble_pred_probs.argmax(axis=1)
    ensemble_preds_mean = ensemble_pred_probs[:,1]
    ensemble_preds_std = ensemble_pred_probs[:,0]*ensemble_pred_probs[:,1]

    std_threshs = np.linspace(np.min(ensemble_preds_std), np.max(ensemble_preds_std), 100)
    reject_rate = [1 - np.mean((ensemble_preds_std<=s)) for s in std_threshs]

    accus = [np.mean((ensemble_preds==y_val_ood)[(ensemble_preds_std<=s)]) for s in std_threshs]
    tps = [np.sum(((y_val_ood)*(ensemble_preds==y_val_ood))[(ensemble_preds_std<=s)]) for s in std_threshs]  # correct and positive
    fps = [np.sum(((ensemble_preds)*(ensemble_preds!=y_val_ood))[(ensemble_preds_std<=s)]) for s in std_threshs]  # incorrect and predicted positive
    AUC = auprc_threshs(ensemble_pred_probs.max(axis=1), y_val_ood, AUCTHRESHS)

    pos = np.sum(y_val_ood)
    recall = [tp/pos for tp in tps]
    precision = [tp/(tp+fp) for tp, fp in zip(tps, fps)]
    
    return precision, recall, AUC


def get_ensemble_preds_from_models(model_pred_probs):
    c1_probs = model_pred_probs.argmax(axis=2).sum(axis=0)/model_pred_probs.shape[0]
    c0_probs = 1 - c1_probs
    pred_probs = np.vstack([c0_probs, c1_probs]).T
    preds = pred_probs.argmax(axis=1)
    return preds, pred_probs

def plot_precision_recall(precisions_df, recalls_df, mp_precision, mp_recall, best_fitness_index={}, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # Create an axis if none is provided
    
    ntrls = recalls_df.shape[1]
    
    # Plotting all Random's
    for i in range(ntrls):
        recall = recalls_df.iloc[:,i].values
        precision = precisions_df.iloc[:,i].values
        ax.plot(recall, precision, marker='+', c='gray', alpha=0.1)

    # Plot full model pool
    ax.plot(mp_recall, mp_precision, marker='+', c='red', label='Model Pool')

    # Plot the best fitness found for each fitness function
    if best_fitness_index:
        for key, value in best_fitness_index.items():
            ax.plot(recalls_df.iloc[:,value].values,
                    precisions_df.iloc[:,value].values,
                    marker='+',
                    label=key)

    # Customize ticks and grid
    ax.set_xticks(np.arange(0, 1.01, step=0.1))
    ax.set_xticks(np.arange(0, 1.01, step=0.05), minor=True)
    ax.set_yticks(np.arange(0.6, 1.01, step=0.05))
    ax.set_ylim(0.6, 1.01)
    ax.grid(True, which='both')

    # Set labels and title
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (Gray are Random)')

    # Add legend
    ax.legend(loc='best')
    
    return ax

def plot_aroc_at_curve(AUCTHRESHS, aucs_df, mp_aucs, best_fitness_index={}, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # Create an axis if none is provided
    
    ntrls = aucs_df.shape[1]

    # Plot all random trials
    for i in range(0, ntrls):
        ax.plot(AUCTHRESHS, aucs_df.iloc[:, i], marker='+', c='gray', alpha=0.2)

    # Plot mean of random trials and full model pool
    ax.plot(AUCTHRESHS, aucs_df.mean(axis=1), c='lime', label='Random Mean')
    ax.plot(AUCTHRESHS, mp_aucs, c='brown', label='Full Model Pool')

    # Plot the best fitness found for each fitness function
    if best_fitness_index:
        for key, value in best_fitness_index.items():
            ax.plot(AUCTHRESHS, aucs_df.iloc[:, value].values, marker='+', label=key)

    # Customize ticks and grid
    ax.set_xticks(np.arange(0, 1.01, step=0.1))
    ax.set_xticks(np.arange(0, 1.01, step=0.05), minor=True)
    
    # Set labels and grid
    ax.set_xlabel('T')
    ax.set_ylabel('AUPRC@Recall<=T')
    ax.set_title('AUC@Recall (Gray are Random)')

    ax.grid(True, which='both')

    # Add legend
    ax.legend(loc='best')

    return ax

def fitness_scatter(fitness_df, aucs_df, col, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # Create a new axis if none is provided

    # Create scatter plots for different thresholds
    ax.scatter(fitness_df[col], aucs_df.iloc[0, :], marker='.', label='T=.1')
    ax.scatter(fitness_df[col], aucs_df.iloc[1, :], marker='.', label='T=.2')

    # Set labels
    ax.set_xlabel(f'{col} fitness')
    ax.set_ylabel('AUPRC@Recall<=T')

    # Add legend
    ax.legend(loc='best')
    
    return ax