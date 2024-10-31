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
    parser.add_argument("-mps", "--num_classifiers", type=int, default=10000, help='Model Pool Size')
    parser.add_argument("-ff", "--feature_fraction", type=float, default=0.5, help='Fraction of features to use for training')
    parser.add_argument("-df", "--data_fraction", type=float, default=0.6, help='Fraction of data to use for training')
    
    parser.add_argument("-md", "--max_depth", type=int, default=10, help='Max depth of DTs')
    parser.add_argument("-msl", "--min_samples_leaf", type=int, default=5, help='Min samples leaf of DTs')
    parser.add_argument("-rs", "--random_state", type=int, default=1, help='Random state')
    parser.add_argument("-ss", "--sample_size", type=int, default=100000, help='Random state')

    parser.add_argument('--clusters_list', nargs='+', type=int, default=[5], help='List of cluster values')
    parser.add_argument("-sfc", "--shift_feature_count", type=int, default=5, help='Number of features to perturb with random noise')

    parser.add_argument("-dp", "--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Reliable Machine Learning/tableshift_datasets", help='Path to dataset')
    parser.add_argument("-dn", "--dataset_name", type=str, default="college_scorecard", help='Dataset Name')

    parser.add_argument("-pn", "--prefix_name", type=str, default='default', help="Path to model pool pickle file")
    
    parser.add_argument("--model_pool_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data/college_scorecard/model_pool.pkl', help="Path to model pool pickle file")
    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data', help='save path')
    parser.add_argument("-sd", "--seed", type=int, default=0, help="random seed")
        
    args = vars(parser.parse_args())
    prefix_name = args['prefix_name']

    # create save directory for experiment
    save_path = args['save_path'] + '/' + args['dataset_name'] + f'/{prefix_name}'
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

    # remove any clusters that are more than data size
    args['clusters_list'] = [x for x in args['clusters_list'] if x <= (min(x_train.shape[0], x_val_id.shape[0])-1)]
    
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

    # 1. Apply Transformation
    x_train, x_val_id = make_noise(x_train, x_val_id, y_train,args['shift_feature_count'], prefix_name)
    x_train = x_train.astype(np.float16)
    x_val_id = x_val_id.astype(np.float16)

    x_train = np.clip(x_train, float16_min, float16_max)
    x_val_id = np.clip(x_val_id, float16_min, float16_max)



    ### 2. Get Cluster dict
    all_clusters_dict = {}
    all_clusters_dict[f'train_{prefix_name}'], all_clusters_dict[f'val_id_{prefix_name}'] = get_clusters_dict(x_train, x_val_id, args['clusters_list'], save_path + f'/clusters_dict.pkl')

    #### Label Shift
    y_train_flipped = DataNoiseAdder.label_flip(y_train)
    y_val_flipped = DataNoiseAdder.label_flip(y_val_id)


    print("Start random search")
    for label_flip in [0, 1]:
        print(f'label_flip:{label_flip}')

        for method in tqdm.tqdm(['train', 'val_id']):
            print(f'method:{method}')
            first_iteration = True

            precisions_df = pd.DataFrame()
            recalls_df = pd.DataFrame()
            aucs_df = pd.DataFrame()
            clusters_dict = all_clusters_dict[f'{method}_{prefix_name}']

            # make preds
            if method == 'train':
                model_pool_pred_probs = model_pool.get_individual_probabilities(x_train).astype(np.float16)
                model_pool_preds = model_pool_pred_probs.argmax(axis=-1).astype(np.float16)
            else:
                model_pool_pred_probs = model_pool.get_individual_probabilities(x_val_id).astype(np.float16)
                model_pool_preds = model_pool_pred_probs.argmax(axis=-1).astype(np.float16)

            for trial in tqdm.tqdm(range(args['ntrls'])):
                tmp = {'generation': trial}

                rnd = np.random.RandomState(trial)
                indices = rnd.choice(model_pool.num_classifiers, size=args['ensemble_size'], replace=True)
                
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

                # Compute precision, recall, and AUC
                if (label_flip == 0)&(method=='train'):
                    precision, recall, auc = get_precision_recall_auc(ood_pred_probs, y_val_ood, AUCTHRESHS)
                    
                    recalls_df = pd.concat([recalls_df, pd.DataFrame(recall)], axis=1)
                    precisions_df = pd.concat([precisions_df, pd.DataFrame(precision)], axis=1)
                    aucs_df = pd.concat([aucs_df, pd.DataFrame(auc)], axis=1)

                    precisions_df.to_csv(save_path +'/precisions_df.csv', index=False)
                    recalls_df.to_csv(save_path +'/recalls_df.csv', index=False)
                    aucs_df.to_csv(save_path +'/aucs_df.csv', index=False)



                if method == 'train':
                    Y = y_train_flipped if label_flip else y_train
                else:
                    Y = y_val_flipped if label_flip else y_val_id
                
                prefix = prefix_name + '_flipped' if label_flip else prefix_name

                model_pred_probs = model_pool_pred_probs[indices]
                model_preds = model_pred_probs.argmax(axis=-1).astype(np.float16)

                # Get ensemble predictions and metrics
                ensemble_preds, ensemble_pred_probs = get_ensemble_preds_from_models(model_pred_probs)
                metrics = EnsembleMetrics(Y, ensemble_preds, ensemble_pred_probs[:, 1])
                diversity = EnsembleDiversity(Y, model_preds)
                
                # Collect metrics in the `tmp` dict (specific to this prefix)
                tmp.update({
                    'ood_acc': ood_accuracy,
                    'ood_auc': ood_auc,
                    'ood_prec': ood_precision,
                    'ood_rec': ood_recall,
                    f'{prefix}_{method}_acc': metrics.accuracy(),
                    f'{prefix}_{method}_auc': metrics.auc(),
                    f'{prefix}_{method}_prec': metrics.precision(),
                    f'{prefix}_{method}_rec': metrics.recall(),
                    f'{prefix}_{method}_f1': metrics.f1(),
                    f'{prefix}_{method}_mae': metrics.mean_absolute_error(),
                    f'{prefix}_{method}_mse': metrics.mean_squared_error(),
                    f'{prefix}_{method}_logloss': metrics.log_loss(),
                    # Diversity metrics
                    f'{prefix}_{method}_q_statistic': np.mean(diversity.q_statistic()),
                    f'{prefix}_{method}_correlation_coefficient': np.mean(diversity.correlation_coefficient()),
                    f'{prefix}_{method}_entropy': np.mean(diversity.entropy()),
                    f'{prefix}_{method}_diversity_measure': diversity.diversity_measure(),
                    f'{prefix}_{method}_hamming_distance': np.mean(diversity.hamming_distance()),
                    f'{prefix}_{method}_error_rate': np.mean(diversity.error_rate()),
                    f'{prefix}_{method}_brier_score': np.mean(diversity.brier_score()),
                    f'{prefix}_{method}_ensemble_variance': np.mean(diversity.ensemble_variance())
                })
                del metrics, diversity
                gc.collect()

                # Compute cluster metrics
                tmp_cluster = compute_cluster_metrics(clusters_dict, ensemble_preds, model_preds, model_pred_probs, Y)

                col_names = [prefix + '_' + x for x in tmp_cluster.columns]
                col_names = [name.replace('_val_acc', '') for name in col_names]
                col_names = [name.replace('_train_acc', '') for name in col_names]
                tmp_cluster.columns = col_names

                # Clean up memory
                del ensemble_preds, ensemble_pred_probs
                gc.collect()

                # save fitness df
                pd.concat([pd.DataFrame([tmp]), tmp_cluster], axis=1).to_csv(save_path +f'/{method}_{label_flip}_fitness_df.csv', mode='a', header=first_iteration, index=False)
                first_iteration = False
                del tmp, tmp_cluster
                gc.collect()  

            if (label_flip == 0)&(method=='train'):
                del precisions_df, recalls_df, aucs_df
                gc.collect()
              


