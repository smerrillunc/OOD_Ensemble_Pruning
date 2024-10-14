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

from utils import get_dataset, get_ensemble_preds_from_models, get_precision_recall_auc, auprc_threshs
from utils import plot_precision_recall, plot_aroc_at_curve, fitness_scatter
from utils import compute_metrics_in_buckets, flatten_df, compute_cluster_metrics
from utils import get_categorical_and_float_features
from utils import create_directory_if_not_exists, save_dict_to_file
from utils import get_clusters_dict

import tqdm
import argparse
from datetime import date

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read file content.')

    parser.add_argument("-n", "--ntrls", type=int, default=100000, help='Number of random search trials')
    parser.add_argument("-e", "--ensemble_size", type=int, default=100, help='Size of ensemble to search for')
    parser.add_argument("-mps", "--num_classifiers", type=int, default=10000, help='Model Pool Size')
    parser.add_argument("-ff", "--feature_fraction", type=float, default=0.5, help='Fraction of features to use for training')
    parser.add_argument("-df", "--data_fraction", type=float, default=0.8, help='Fraction of data to use for training')
    
    parser.add_argument("-md", "--max_depth", type=int, default=10, help='Max depth of DTs')
    parser.add_argument("-msl", "--min_samples_leaf", type=int, default=5, help='Min samples leaf of DTs')
    parser.add_argument("-rs", "--random_state", type=int, default=1, help='Random state')

    parser.add_argument("-sfc", "--shift_feature_count", type=int, default=5, help='Number of features to perturb with random noise')

    parser.add_argument("-dp", "--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/datasets", help='Path to dataset')
    parser.add_argument("-dn", "--dataset_name", type=str, default="heloc_tf", help='Dataset Name')



    parser.add_argument("--save_name", type=str, default=None, help="Save Name")

    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data', help='save path')
        
    args = vars(parser.parse_args())

    if args['save_name'] == None:
        args['save_name'] = date.today().strftime('%Y%m%d')

    # create save directory for experiment
    save_path = create_directory_if_not_exists(args['save_path'] + '/' + args['save_name'])
    save_dict_to_file(args, save_path + '/experiment_args.txt')

    args['clusters_list'] = [3, 10, 25, 100]
    args['shift_feature_count'] = 5

    AUCTHRESHS = np.array([0.1, 0.2, 0.3, 0.4, 1. ])

    x_train, y_train, x_val_id, y_val_id, x_val_ood, y_val_ood = get_dataset(args['dataset_path'] , args['dataset_name'])
    num_features = x_train.shape[1]


    ### Building and Training Model Pool
    model_pool = DecisionTreeEnsemble(args['num_classifiers'], 
                                      args['feature_fraction'],
                                      args['data_fraction'],
                                      args['max_depth'],
                                      args['min_samples_leaf'],
                                      args['random_state'])

    model_pool.train(x_train, y_train)


    # ### Caching Model Pool Predictions
    model_pool_preds = model_pool.predict(x_val_ood)
    model_pool_pred_probs = model_pool.predict_proba(x_val_ood)
    mp_precision, mp_recall, mp_auc = get_precision_recall_auc(model_pool_pred_probs, y_val_ood, AUCTHRESHS)


    # ### Caching Individual Model Predictions
    model_pool.train_preds = model_pool.get_individual_predictions(x_train).T
    model_pool.train_pred_probs = model_pool.get_individual_probabilities(x_train)

    model_pool.val_id_preds = model_pool.get_individual_predictions(x_val_id).T
    model_pool.val_id_pred_probs = model_pool.get_individual_probabilities(x_val_id)

    model_pool.val_ood_preds = model_pool.get_individual_predictions(x_val_ood).T
    model_pool.val_ood_pred_probs = model_pool.get_individual_probabilities(x_val_ood)


    # ### Clustering Data into different groupings and preparing formating
    all_clusters_dict = {}
    all_clusters_dict['train_preds'], all_clusters_dict['val_id']  = get_clusters_dict(x_train, x_val_id, args['clusters_list'])
    all_clusters_dict = make_noise_preds(x_train, y_train, x_val_id, model_pool, args['shift_feature_count'], args['clusters_list'], all_clusters_dict)

    # ### Label Shift
    y_train_flipped = DataNoiseAdder.label_flip(y_train)
    y_val_flipped = DataNoiseAdder.label_flip(y_val_id)

    # ### Random Search Loop
    a = [x for x in dir(model_pool) if 'preds' in x]
    b = [x for x in dir(model_pool) if 'pred_' in x]
    a.remove('val_ood_preds')
    b.remove('val_ood_pred_probs')
    pred_attributes = [(a[i], b[i], a[i].split('_')[0]+'_'+a[i].split('_')[1]) for i in range(len(a))]

    precisions_df = pd.DataFrame()
    recalls_df = pd.DataFrame()
    aucs_df = pd.DataFrame()
    fitness_df = pd.DataFrame()

    print("Start random search")
    for trial in tqdm.tqdm(range(args['ntrls'])):
        indices = np.random.choice(model_pool.num_classifiers, size=args['ensemble_size'], replace=True)

        # ood preds of sub-ensemble
        ood_preds, ood_pred_probs = get_ensemble_preds_from_models(model_pool.val_ood_pred_probs[indices])

        # save OOD precision/recalls seprately
        precision, recall, auc = get_precision_recall_auc(ood_pred_probs, y_val_ood, AUCTHRESHS)

        recalls_df = pd.concat([recalls_df, pd.DataFrame(recall)], axis=1)
        precisions_df = pd.concat([precisions_df, pd.DataFrame(precision)], axis=1)
        aucs_df = pd.concat([aucs_df, pd.DataFrame(auc)], axis=1)

        tmp = {'generation':trial,
                  'ensemble_files':','.join(str(x) for x in indices)}
        cluster_metrics = pd.DataFrame()

        # Compute all Fitness Metrics
        for label_flip in [0, 1]:
            for pred_tuple in pred_attributes:
                preds_name, pred_prob_name, prefix_name = pred_tuple
    
                # get clustering associated with data transformation
                clusters_dict = all_clusters_dict[prefix_name]
                if 'train' in prefix_name:
                    if label_flip:
                        Y = y_train_flipped
                        prefix_name = prefix_name + '_flip'
                    else:
                        Y = y_train
                        
                else:
                    if label_flip:
                        prefix_name = prefix_name + '_flip'
                        Y = y_val_flipped
                    else:
                        Y = y_val_id

                model_pool_preds = getattr(model_pool, preds_name)
                model_pool_pred_probs = getattr(model_pool, pred_prob_name)

                model_preds = model_pool_preds[indices]
                model_pred_probs = model_pool_pred_probs[indices]

                # id val preds of sub-ensemble
                ensemble_preds, ensemble_pred_probs = get_ensemble_preds_from_models(model_pred_probs)
                metrics = EnsembleMetrics(Y, ensemble_preds, ensemble_pred_probs[:,1])
                diversity = EnsembleDiversity(Y, model_preds)

                tmp.update({f'{prefix_name}_acc':metrics.accuracy(),
                       f'{prefix_name}_auc':metrics.auc(),
                       f'{prefix_name}_prec':metrics.precision(),
                       f'{prefix_name}_rec':metrics.recall(),
                       f'{prefix_name}_f1':metrics.f1(),
                       f'{prefix_name}_mae':metrics.mean_absolute_error(),
                       f'{prefix_name}_mse':metrics.mean_squared_error(),
                       f'{prefix_name}_logloss':metrics.log_loss(),

                       # diversity
                       f'{prefix_name}_q_statistic':np.mean(diversity.q_statistic()),
                       f'{prefix_name}_correlation_coefficient':np.mean(diversity.correlation_coefficient()),
                       f'{prefix_name}_entropy':np.mean(diversity.entropy()),
                       f'{prefix_name}_diversity_measure':diversity.diversity_measure(),
                       f'{prefix_name}_hamming_distance':np.mean(diversity.hamming_distance()),
                       f'{prefix_name}_error_rate':np.mean(diversity.error_rate()),
                       f'{prefix_name}_auc':np.mean(diversity.auc()),
                       f'{prefix_name}_brier_score':np.mean(diversity.brier_score()),
                       f'{prefix_name}_ensemble_variance':np.mean(diversity.ensemble_variance()),
                      })

                # compute cluster metrics
                tmp_cluster = compute_cluster_metrics(clusters_dict, ensemble_preds, model_preds, model_pred_probs, Y)
                col_names = [prefix_name + '_' + x for x in tmp_cluster.columns]
                col_names = [name.replace('_val_acc', '') for name in col_names]
                col_names = [name.replace('_train_acc', '') for name in col_names]
                tmp_cluster.columns = col_names

                cluster_metrics = pd.concat([cluster_metrics, tmp_cluster], axis=1)

        raw_metrics = pd.DataFrame([tmp])    
        tmp = pd.concat([raw_metrics, cluster_metrics], axis=1)
        fitness_df = pd.concat([fitness_df, tmp])
        precisions_df.to_csv(save_path+'/precisions_df.csv', index=False)
        recalls_df.to_csv(save_path+'/recalls_df.csv', index=False)
        aucs_df.to_csv(save_path+'/aucs_df.csv', index=False)
        fitness_df.to_csv(save_path+'/fitness_df.csv', index=False)
        