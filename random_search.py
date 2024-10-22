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
from utils import get_clusters_dict, make_noise_preds
from utils import create_directory_if_not_exists, save_dict_to_file

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
    
    parser.add_argument("--model_pool_path", type=str, default=None, help="Path to model pool pickle file")
    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data', help='save path')
    parser.add_argument("-sd", "--seed", type=int, default=0, help="random seed")
        
    args = vars(parser.parse_args())

    # create save directory for experiment
    save_path = args['save_path'] + '/' + args['dataset_name']
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/fitness_metrics', exist_ok=True)

    save_dict_to_file(args, save_path + '/experiment_args.txt')


    AUCTHRESHS = np.array([0.1, 0.2, 0.3, 0.4, 1. ])

    rnd = np.random.RandomState(args['seed'])
    x_train, y_train, x_val_id, y_val_id, x_val_ood, y_val_ood = get_tableshift_dataset(args['dataset_path'] , args['dataset_name'])
    sample_indices_train = rnd.choice(x_train.shape[0], size=min(x_train.shape[0], args['sample_size']), replace=True)
    x_train = x_train[sample_indices_train]
    y_train = y_train[sample_indices_train]

    sample_indices_val = rnd.choice(x_val_id.shape[0], size=min(x_val_id.shape[0], args['sample_size']), replace=True)
    x_val_id = x_val_id[sample_indices_val]
    y_val_id = y_val_id[sample_indices_val]

    sample_indices_ood = rnd.choice(x_val_ood.shape[0], size=min(x_val_ood.shape[0], args['sample_size']), replace=True)
    x_val_ood = x_val_ood[sample_indices_ood]
    y_val_ood = y_val_ood[sample_indices_ood]

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
        model_pool = model_pool.load(args['model_pool_path'])

    else:
        ### Building and Training Model Pool
        print('Building and Training Model Pool')


        model_pool = DecisionTreeEnsemble(args['num_classifiers'], 
                                          args['feature_fraction'],
                                          args['data_fraction'],
                                          args['max_depth'],
                                          args['min_samples_leaf'],
                                          args['random_state'])

        model_pool.train(x_train, y_train)
        print('saving_model_pool to ')
        model_pool.save(save_path + '/model_pool.pkl')

    
    # ###  Model Pool Predictions
    model_pool_preds = model_pool.predict(x_val_ood)
    model_pool_pred_probs = model_pool.predict_proba(x_val_ood)
    mp_precision, mp_recall, mp_auc = get_precision_recall_auc(model_pool_pred_probs, y_val_ood, AUCTHRESHS)
    
    with open(save_path + f'/model_pool_precision.pkl', 'wb') as file:
        pickle.dump(mp_precision, file)

    with open(save_path + f'/model_pool_recall.pkl', 'wb') as file:
        pickle.dump(mp_recall, file)

    with open(save_path + f'/model_pool_auc.pkl', 'wb') as file:
        pickle.dump(mp_auc, file)


    # ### Saving Individual Model Predictions
    np.save(save_path + '/train_pred_probs.npy', model_pool.get_individual_probabilities(x_train))
    np.save(save_path + '/val_id_pred_probs.npy', model_pool.get_individual_probabilities(x_val_id))

    # cache these
    model_pool.val_ood_pred_probs = model_pool.get_individual_probabilities(x_val_ood)


    # ### Clustering Data into different groupings and preparing formating
    all_clusters_dict = {}
    all_clusters_dict['train'], all_clusters_dict['val_id']  = get_clusters_dict(x_train, x_val_id, args['clusters_list'], save_path + '/default.pkl')
    all_clusters_dict = make_noise_preds(x_train, y_train, x_val_id, model_pool, args['shift_feature_count'], args['clusters_list'], all_clusters_dict, save_path)

    generator = SyntheticDataGenerator(x_train, y_train)

    # interpolation
    interp_x, interp_y = generator.interpolate(x_train.shape[0])
    np.save(save_path + '/synth_interp_pred_probs.npy', model_pool.get_individual_probabilities(interp_x))
    del interp_x

    # GMM
    gmm_x, gmm_y = generator.gaussian_mixture(x_train.shape[0])
    np.save(save_path + '/synth_gmm_pred_probs.npy', model_pool.get_individual_probabilities(gmm_x))

    del gmm_x

    dt_x, dt_y = generator.decision_tree(x_train.shape[0])
    np.save(save_path + '/synth_dt_pred_probs.npy', model_pool.get_individual_probabilities(dt_x))
    del dt_x

    synth_data_dict = {'synth_interp':interp_y.astype('int64'),
                       'synth_gmm': gmm_y.astype('int64'),
                       'synth_dt': dt_y.astype('int64')}
    # ### Label Shift
    y_train_flipped = DataNoiseAdder.label_flip(y_train)
    y_val_flipped = DataNoiseAdder.label_flip(y_val_id)

    # ### Random Search Loop
    prefix_names = ['train', 'val_id',
                    'train_gaussian', 'val_gaussian',
                    'train_uniform', 'val_uniform',
                    'train_laplace', 'val_laplace',
                    'train_dropout', 'val_dropout',
                    'train_boundaryshift', 'val_boundaryshift',
                    'train_upscaleshift', 'val_upscaleshift',
                    'train_downscaleshift', 'val_downscaleshift',
                    'train_distshiftuniform', 'val_distshiftuniform',
                    'train_distshiftgaussian', 'val_distshiftgaussian',
                    'synth_interp', 'synth_gmm', 'synth_dt']
    from memory_profiler import profile

    @profile
    def run_random_search():
        print("Start random search")
        
        for trial in tqdm.tqdm(range(args['ntrls'])):
            indices = np.random.choice(model_pool.num_classifiers, size=args['ensemble_size'], replace=True)
            
            # Get ensemble predictions
            ood_preds, ood_pred_probs = get_ensemble_preds_from_models(model_pool.val_ood_pred_probs[indices])
            
            # Compute precision, recall, and AUC
            precision, recall, auc = get_precision_recall_auc(ood_pred_probs, y_val_ood, AUCTHRESHS)
            
            # Append precision, recall, and AUC to corresponding CSV files
            pd.DataFrame(precision).to_csv(save_path + '/precisions_df.csv', mode='a', header=False, index=False)
            pd.DataFrame(recall).to_csv(save_path + '/recalls_df.csv', mode='a', header=False, index=False)
            pd.DataFrame(auc).to_csv(save_path + '/aucs_df.csv', mode='a', header=False, index=False)
            
            # Simplified string join, reduced memory usage
            ensemble_files_str = ','.join(map(str, indices))
            base_tmp = {'generation': trial, 'ensemble_files': ensemble_files_str}
            
            for label_flip in [0]:  # [0, 1]
                print('Evaluating Prefixes')
                for prefix_name in tqdm.tqdm(prefix_names):
                    print(f'{prefix_name}, {label_flip}')
                    
                    if 'synth' not in prefix_name:
                        try:
                            clusters_dict = all_clusters_dict[prefix_name]
                        except Exception as e:
                            print(e)
                    
                    if 'train' in prefix_name:
                        Y = y_train_flipped if label_flip else y_train
                    elif 'synth' in prefix_name:
                        if label_flip:
                            continue
                        else:
                            Y = synth_data_dict[prefix_name]
                    else:
                        Y = y_val_flipped if label_flip else y_val_id
                    
                    try:
                        model_pool_pred_probs = np.load(save_path + f'/{prefix_name}_pred_probs.npy')
                        model_pool_preds = model_pool_pred_probs.argmax(axis=-1)
                    except Exception as e:
                        print(e)

                    model_pred_probs = model_pool_pred_probs[indices]
                    model_preds = model_pred_probs.argmax(axis=-1)

                    # Get ensemble predictions and metrics
                    ensemble_preds, ensemble_pred_probs = get_ensemble_preds_from_models(model_pred_probs)
                    metrics = EnsembleMetrics(Y, ensemble_preds, ensemble_pred_probs[:, 1])
                    diversity = EnsembleDiversity(Y, model_preds)
                    
                    # Collect metrics in the `tmp` dict (specific to this prefix)
                    tmp = base_tmp.copy()
                    tmp.update({
                        f'{prefix_name}_acc': metrics.accuracy(),
                        f'{prefix_name}_auc': metrics.auc(),
                        f'{prefix_name}_prec': metrics.precision(),
                        f'{prefix_name}_rec': metrics.recall(),
                        f'{prefix_name}_f1': metrics.f1(),
                        f'{prefix_name}_mae': metrics.mean_absolute_error(),
                        f'{prefix_name}_mse': metrics.mean_squared_error(),
                        f'{prefix_name}_logloss': metrics.log_loss(),
                        # Diversity metrics
                        f'{prefix_name}_q_statistic': np.mean(diversity.q_statistic()),
                        f'{prefix_name}_correlation_coefficient': np.mean(diversity.correlation_coefficient()),
                        f'{prefix_name}_entropy': np.mean(diversity.entropy()),
                        f'{prefix_name}_diversity_measure': diversity.diversity_measure(),
                        f'{prefix_name}_hamming_distance': np.mean(diversity.hamming_distance()),
                        f'{prefix_name}_error_rate': np.mean(diversity.error_rate()),
                        f'{prefix_name}_brier_score': np.mean(diversity.brier_score()),
                        f'{prefix_name}_ensemble_variance': np.mean(diversity.ensemble_variance())
                    })
                    
                    # Compute cluster metrics
                    tmp_cluster = compute_cluster_metrics(clusters_dict, ensemble_preds, model_preds, model_pred_probs, Y)
                    
                    # Clean up memory
                    del ensemble_preds, ensemble_pred_probs
                    gc.collect()

                    # Batch processing column name replacement
                    col_names = [prefix_name + '_' + x for x in tmp_cluster.columns]
                    col_names = [name.replace('_val_acc', '').replace('_train_acc', '') for name in col_names]
                    tmp_cluster.columns = col_names
                    
                    # Append cluster metrics to a CSV
                    tmp_cluster.to_csv(save_path + '/fitness_metrics/{prefix_name}_cluster.csv', mode='a', header=False, index=False)
                    
                    # Save the current tmp dictionary to a file specific to the prefix
                    raw_metrics = pd.DataFrame([tmp])
                    raw_metrics.to_csv(save_path + f'/fitness_metrics/{prefix_name}.csv', mode='a', header=False, index=False)

        print("Random search complete")
        return None



    _ = run_random_search()
    precisions_df.to_csv(save_path+'/precisions_df.csv', index=False)

        