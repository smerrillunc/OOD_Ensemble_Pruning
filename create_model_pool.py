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
    parser.add_argument("-ff", "--feature_fraction", type=float, default=0.5, help='Fraction of features to use for training')
    parser.add_argument("-df", "--data_fraction", type=float, default=0.6, help='Fraction of data to use for training')
    
    parser.add_argument("-n", "--num_classifiers", type=int, default=10000, help='Max depth of DTs')
    parser.add_argument("-md", "--max_depth", type=int, default=10, help='Max depth of DTs')
    parser.add_argument("-msl", "--min_samples_leaf", type=int, default=5, help='Min samples leaf of DTs')
    parser.add_argument("-rs", "--random_state", type=int, default=1, help='Random state')
    parser.add_argument("-ss", "--sample_size", type=int, default=100000, help='Random state')

    parser.add_argument("-dp", "--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Reliable Machine Learning/tableshift_datasets", help='Path to dataset')
    parser.add_argument("-dn", "--dataset_name", type=str, default="college_scorecard", help='Dataset Name')

    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data', help='save path')
    parser.add_argument("-sd", "--seed", type=int, default=0, help="random seed")
        
    args = vars(parser.parse_args())

    # create save directory for experiment
    save_path = args['save_path'] + '/' + args['dataset_name']
    os.makedirs(save_path, exist_ok=True)

    save_dict_to_file(args, save_path + '/experiment_args.txt')


    AUCTHRESHS = np.array([0.1, 0.2, 0.3, 0.4, 1. ])

    rnd = np.random.RandomState(args['seed'])
    x_train, y_train, _, _, _, _ = get_tableshift_dataset(args['dataset_path'] , args['dataset_name'])
    sample_indices_train = rnd.choice(x_train.shape[0], size=min(x_train.shape[0], args['sample_size']), replace=True)
    x_train = x_train[sample_indices_train]
    y_train = y_train[sample_indices_train]

    assert(len(np.unique(y_train)) == 2)
    num_features = x_train.shape[1]

    x_train = x_train.astype(np.float16)

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