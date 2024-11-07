#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Clustering import Clustering
from DataNoiseAdder import DataNoiseAdder
from DatasetCorruptor import DatasetCorruptor
from DecisionTreeEnsemble import DecisionTreeEnsemble
from SyntheticDataGenerator import SyntheticDataGenerator
from EnsembleDiversity import EnsembleDiversity
from EnsembleMetrics import EnsembleMetrics
from OutlierDetector import OutlierDetector

from utils import get_tableshift_dataset, get_ensemble_preds_from_models, get_precision_recall_auc, auprc_threshs
from utils import plot_precision_recall, plot_aroc_at_curve, fitness_scatter
from utils import compute_metrics_in_buckets, flatten_df, compute_cluster_metrics
from utils import get_categorical_and_float_features
from utils import get_clusters_dict, make_noise
from utils import create_directory_if_not_exists, save_dict_to_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import tqdm
import argparse
from datetime import date

import warnings
import pickle
import gc
import os

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read file content.')

    parser.add_argument("-n", "--ntrls", type=int, default=100000, help='Number of random search trials')
    parser.add_argument("-e", "--ensemble_size", type=int, default=100, help='Size of ensemble to search for')
    parser.add_argument("-ss", "--sample_size", type=int, default=100000, help='Random state')

    parser.add_argument("-dp", "--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Reliable Machine Learning/tableshift_datasets", help='Path to dataset')
    parser.add_argument("-dn", "--dataset_name", type=str, default="college_scorecard", help='Dataset Name')
    parser.add_argument("--indices_file", type=str, default=None, help="Indices to test")

    parser.add_argument("--model_pool_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data/college_scorecard/model_pool.pkl', help="Path to model pool pickle file")
    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data', help='save path')
    parser.add_argument("-sd", "--seed", type=int, default=0, help="random seed")
        
    args = vars(parser.parse_args())

    # create save directory for experiment
    save_path = args['save_path'] + '/' + args['dataset_name'] + f'/outliers'
    os.makedirs(save_path, exist_ok=True)
    
    save_dict_to_file(args, save_path + '/experiment_args.txt')

    AUCTHRESHS = np.array([0.1, 0.2, 0.3, 0.4, 1. ])
    float16_min = np.finfo(np.float16).min  # Smallest (most negative) float16 value    
    float16_max = np.finfo(np.float16).max  # Largest float16 value

    rnd = np.random.RandomState(args['seed'])
    x_train, y_train, x_val_id, y_val_id, x_val_ood, y_val_ood = get_tableshift_dataset(args['dataset_path'] , args['dataset_name'])
    sample_indices_train = rnd.choice(x_train.shape[0], size=min(x_train.shape[0], args['sample_size']), replace=False)
    x_train = x_train[sample_indices_train]
    y_train = y_train[sample_indices_train]

    sample_indices_val = rnd.choice(x_val_id.shape[0], size=min(x_val_id.shape[0], args['sample_size']), replace=False)
    x_val_id = x_val_id[sample_indices_val]
    y_val_id = y_val_id[sample_indices_val]

    x_val_ood = x_val_ood
    y_val_ood = y_val_ood

    assert(len(np.unique(y_train)) == 2)
    assert(len(np.unique(y_val_id)) == 2)
    assert(len(np.unique(y_val_ood)) == 2)

    num_features = x_train.shape[1]

    x_train = x_train.astype(np.float16)
    x_val_id = x_val_id.astype(np.float16)
    x_val_ood = x_val_ood.astype(np.float16)

    if args['model_pool_path']:
        # loading model pool from file
        print('loading model pool from file')
        model_pool = DecisionTreeEnsemble.load(args['model_pool_path'])
    else:
        print("Invalid model pool path")

    # cache ood preds
    model_pool.val_ood_pred_probs = model_pool.get_individual_probabilities(x_val_ood)
    model_pool.val_ood_pred_probs = model_pool.val_ood_pred_probs.astype(np.float16)
    del x_val_ood
    gc.collect()

    categorical_features, float_features= get_categorical_and_float_features(x_train)

    # Usage
    detector = OutlierDetector(x_train[:,categorical_features], y_train)

    # Get outliers using different methods
    freq_indices = detector.frequency_based_outliers()
    iso_forest_indices = detector.isolation_forest_outliers()
    lof_indices = detector.lof_outliers()
    dbscan_indices = detector.dbscan_outliers(eps=0.5, min_samples=5)
    logistic_indices = detector.logistic_regression_outliers()

    np.savez(save_path + '/outliers.npz', 
         freq_indices=freq_indices, 
         iso_forest_indices=iso_forest_indices,
         lof_indices=lof_indices,
         dbscan_indices=dbscan_indices,
         logistic_indices=logistic_indices)



    outlier_sets = [('outlier_freq', freq_indices),
                    ('outlier_iso', iso_forest_indices),
                    ('outlier_lof', lof_indices),
                    ('outlier_dbscan', dbscan_indices),
                    ('outlier_logistic', logistic_indices)]

    precisions_df = pd.DataFrame()
    recalls_df = pd.DataFrame()
    aucs_df = pd.DataFrame()


    if args['indices_file']:
        print('loading indices file')
        # will search for these indices
        tmp = pd.read_csv(args['indices_file'])
        indices_list = tmp.drop_duplicates(subset=['Best_Ensemble_Indices'])['Best_Ensemble_Indices'].values
        args['ntrls'] = len(indicies)


    model_pool_pred_probs = model_pool.get_individual_probabilities(x_train).astype(np.float16)
    first_iteration = True
    for trial in tqdm.tqdm(range(args['ntrls'])):
        tmp = {'generation': trial}

        for prefix, outliers in outlier_sets:
            rnd = np.random.RandomState(trial)
            
            if args['ensemble_size']:
                indices = rnd.choice(model_pool.num_classifiers, size=args['ensemble_size'], replace=True)
            elif args['indices_file']:
                indices = indices_list[trial]
            else:
                ensemble_size = rnd.randint(10, model_pool.num_classifiers*0.1)
                indices = rnd.choice(model_pool.num_classifiers, size=ensemble_size, replace=True)            

            # Get ensemble predictions
            ood_preds, ood_pred_probs = get_ensemble_preds_from_models(model_pool.val_ood_pred_probs[indices])
            ood_preds = ood_preds.astype(np.float16)
            ood_pred_probs = ood_pred_probs.astype(np.float16)

            ood_accuracy = accuracy_score(y_val_ood, ood_preds)
            ood_precision = precision_score(y_val_ood, ood_preds)
            ood_recall = recall_score(y_val_ood, ood_preds)

            try:
                ood_auc = roc_auc_score(y_val_ood, ood_pred_probs[:,1])
            except:
                ood_auc = np.nan

            precision, recall, auc = get_precision_recall_auc(ood_pred_probs, y_val_ood, AUCTHRESHS)
            
            recalls_df = pd.concat([recalls_df, pd.DataFrame(recall)], axis=1)
            precisions_df = pd.concat([precisions_df, pd.DataFrame(precision)], axis=1)
            aucs_df = pd.concat([aucs_df, pd.DataFrame(auc)], axis=1)

            precisions_df.to_csv(save_path +'/precisions_df.csv', index=False)
            recalls_df.to_csv(save_path +'/recalls_df.csv', index=False)
            aucs_df.to_csv(save_path +'/aucs_df.csv', index=False)


            model_pred_probs = model_pool_pred_probs[indices][:,outliers,:]
            model_preds = model_pred_probs.argmax(axis=-1).astype(np.float16)

            # Get ensemble predictions and metrics
            ensemble_preds, ensemble_pred_probs = get_ensemble_preds_from_models(model_pred_probs)
            metrics = EnsembleMetrics(y_train[outliers], ensemble_preds, ensemble_pred_probs[:, 1])
            diversity = EnsembleDiversity(y_train[outliers], model_preds)
            
            # Collect metrics in the `tmp` dict (specific to this prefix)
            tmp.update({
                'ood_acc': ood_accuracy,
                'ood_auc': ood_auc,
                'ood_prec': ood_precision,
                'ood_rec': ood_recall,
                f'{prefix}_acc': metrics.accuracy(),
                f'{prefix}_auc': metrics.auc(),
                f'{prefix}_prec': metrics.precision(),
                f'{prefix}_rec': metrics.recall(),
                f'{prefix}_f1': metrics.f1(),
                f'{prefix}_mae': metrics.mean_absolute_error(),
                f'{prefix}_mse': metrics.mean_squared_error(),
                f'{prefix}_logloss': metrics.log_loss(),
                # Diversity metrics
                f'{prefix}_q_statistic': np.mean(diversity.q_statistic()),
                f'{prefix}_correlation_coefficient': np.mean(diversity.correlation_coefficient()),
                f'{prefix}_entropy': np.mean(diversity.entropy()),
                f'{prefix}_diversity_measure': diversity.diversity_measure(),
                f'{prefix}_hamming_distance': np.mean(diversity.hamming_distance()),
                f'{prefix}_error_rate': np.mean(diversity.error_rate()),
                f'{prefix}_brier_score': np.mean(diversity.brier_score()),
                f'{prefix}_ensemble_variance': np.mean(diversity.ensemble_variance())
            })

        # save fitness df
        pd.DataFrame([tmp]).to_csv(save_path +f'/outlier_fitness_df.csv', mode='a', header=first_iteration, index=False)
        first_iteration = False

        del metrics, diversity
        del ensemble_preds, ensemble_pred_probs
        del tmp
        gc.collect()  


