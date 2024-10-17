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

    parser.add_argument('--clusters_list', nargs='+', type=int, default=[5], help='List of cluster values')
    parser.add_argument("-sfc", "--shift_feature_count", type=int, default=5, help='Number of features to perturb with random noise')

    parser.add_argument("-dp", "--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Reliable Machine Learning/tableshift_datasets", help='Path to dataset')
    parser.add_argument("-dn", "--dataset_name", type=str, default="college_scorecard", help='Dataset Name')
    
    parser.add_argument("--save_name", type=str, default=None, help="Save Name")
    parser.add_argument("--model_pool_path", type=str, default=None, help="Path to model pool pickle file")
    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data', help='save path')
        
    args = vars(parser.parse_args())

    if args['save_name'] == None:
        args['save_name'] = date.today().strftime('%Y%m%d')

    # create save directory for experiment
    save_path = create_directory_if_not_exists(args['save_path'] + '/' + args['dataset_name'] + '/' + args['save_name'])
    clusters_save_path = args['save_path'] + '/' +  args['dataset_name'] 
    model_pool_save_path = args['save_path'] + '/' + args['dataset_name'] + '/model_pool.pkl'
    save_dict_to_file(args, save_path + '/experiment_args.txt')


    AUCTHRESHS = np.array([0.1, 0.2, 0.3, 0.4, 1. ])

    x_train, y_train, x_val_id, y_val_id, x_val_ood, y_val_ood = get_tableshift_dataset(args['dataset_path'] , args['dataset_name'])
    assert(len(np.unique(y_train)) == 2)
    assert(len(np.unique(y_val_id)) == 2)
    assert(len(np.unique(y_val_ood)) == 2)
    num_features = x_train.shape[1]

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
        model_pool.save(model_pool_save_path)

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
    all_clusters_dict['train'], all_clusters_dict['val_id']  = get_clusters_dict(x_train, x_val_id, args['clusters_list'], clusters_save_path + '/default.pkl')
    all_clusters_dict = make_noise_preds(x_train, y_train, x_val_id, model_pool, args['shift_feature_count'], args['clusters_list'], all_clusters_dict, clusters_save_path, save_path)

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
        for label_flip in [0]: #[0, 1]:
            print('Evaluating Prefixes')
            for prefix_name in tqdm.tqdm(prefix_names):
                print(f'{prefix_name}, {label_flip}')
                # get clustering associated with data transformation
                if 'synth' not in prefix_name:
                    clusters_dict = all_clusters_dict[prefix_name]
                    
                if 'train' in prefix_name:
                    if label_flip:
                        Y = y_train_flipped
                        prefix_name = prefix_name + '_flip'
                    else:
                        Y = y_train
                        
                elif 'synth' in prefix_name:
                    if label_flip:
                        continue
                    else:
                        Y = synth_data_dict[prefix_name]

                else:
                    if label_flip:
                        prefix_name = prefix_name + '_flip'
                        Y = y_val_flipped
                    else:
                        Y = y_val_id

                model_pool_pred_probs = np.load(save_path + f'/{prefix_name}_pred_probs.npy')
                model_pool_preds = model_pool_pred_probs.argmax(axis=-1)

                model_pred_probs = model_pool_pred_probs[indices]
                model_preds = model_pred_probs.argmax(axis=-1)

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
        