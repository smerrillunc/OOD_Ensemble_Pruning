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
import random

import tqdm
import argparse
from datetime import date

import warnings
import pickle
import gc
import os
import csv

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read file content.')

    parser.add_argument("-p", "--population_size", type=int, default=1000, help='Number of random search trials')
    parser.add_argument("-e", "--ensemble_size", type=int, default=100, help='Size of ensemble to search for')
    parser.add_argument("-g", "--generations", type=int, default=100000, help='Size of ensemble to search for')
    parser.add_argument("-el", "--elitism", type=int, default=200, help='Size of ensemble to search for')
    parser.add_argument("-t", "--tournament_size", type=int, default=500, help='Size of ensemble to search for')
    parser.add_argument("-mr", "--mutation_rate", type=float, default=0.1, help='Size of ensemble to search for')

    parser.add_argument("-dp", "--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Reliable Machine Learning/tableshift_datasets", help='Path to dataset')
    parser.add_argument("-dn", "--dataset_name", type=str, default="college_scorecard", help='Dataset Name')

    
    parser.add_argument("--model_pool_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data/college_scorecard/model_pool.pkl', help="Path to model pool pickle file")
    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/data', help='save path')
    parser.add_argument("-sd", "--seed", type=int, default=0, help="random seed")
        
    args = vars(parser.parse_args())

    # create save directory for experiment
    save_path = args['save_path'] + '/cheating_search/' + args['dataset_name'] + f'/'
    os.makedirs(save_path, exist_ok=True)
    
    save_dict_to_file(args, save_path + '/experiment_args.txt')

    AUCTHRESHS = np.array([0.1, 0.2, 0.3, 0.4, 1. ])

    rnd = np.random.RandomState(args['seed'])
    _, _, _, _, x_val_ood, y_val_ood = get_tableshift_dataset(args['dataset_path'] , args['dataset_name'])
    x_val_ood = x_val_ood
    y_val_ood = y_val_ood

    assert(len(np.unique(y_val_ood)) == 2)

    num_features = x_val_ood.shape[1]
    x_val_ood = x_val_ood.astype(np.float16)
    
    if args['model_pool_path']:
        # loading model pool from file
        print('loading model pool from file')
        model_pool = DecisionTreeEnsemble.load(args['model_pool_path'])
    else:
        print("Invalid model pool path")

    # cache ood preds
    model_pool.val_ood_pred_probs = model_pool.get_individual_probabilities(x_val_ood)

    # Hyperparameters
    POPULATION_SIZE = args['population_size']
    ENSEMBLE_SIZE = args['ensemble_size']
    NUM_GENERATIONS = args['generations']
    TOURNAMENT_SIZE = args['tournament_size']
    MUTATION_RATE = args['mutation_rate']
    ELITISM_SIZE = args['elitism']

    # Fitness function to evaluate OOD accuracy
    def calculate_ood_accuracy(ensemble):
        # Get ensemble predictions
        ood_preds, ood_pred_probs = get_ensemble_preds_from_models(model_pool.val_ood_pred_probs[ensemble])
        acc = (ood_preds == y_val_ood).mean()
        return acc

    # Initialize the population with random ensembles
    def initialize_population(pool_size, ensemble_size, population_size):
        return [random.sample(range(pool_size), ensemble_size) for _ in range(population_size)]

    # Tournament selection
    def tournament_selection(population, scores, tournament_size):
        selected = random.sample(list(zip(population, scores)), tournament_size)
        selected.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness
        return selected[0][0]  # Return the best ensemble

    # One-point crossover
    def crossover(parent1, parent2):
        crossover_point = random.randint(1, ENSEMBLE_SIZE - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    # Mutation
    def mutate(ensemble, pool_size):
        if random.random() < MUTATION_RATE:
            index_to_mutate = random.randint(0, ENSEMBLE_SIZE - 1)
            ensemble[index_to_mutate] = random.randint(0, pool_size - 1)
        return ensemble

    # Evolutionary algorithm
    def evolutionary_algorithm(pool_size):
        # Initialize population
        population = initialize_population(pool_size, ENSEMBLE_SIZE, POPULATION_SIZE)
        
        # Open the CSV file in append mode
        with open(save_path + "/all_ensembles_log.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header once
            writer.writerow(["Generation", "Ensemble_Indices", "OOD_Accuracy"])

            for generation in tqdm.tqdm(range(NUM_GENERATIONS)):
                # Calculate fitness scores for the current generation
                scores = [calculate_ood_accuracy(ensemble) for ensemble in population]
                
                # Log each ensemble with its score in the current generation
                for ensemble, score in zip(population, scores):
                    # Write each ensemble with its score to the file
                    writer.writerow([generation, ensemble[:], score])
                
                # Print the best score for the current generation
                best_score = max(scores)
                print(f"Generation {generation}, Best Score: {best_score}")

                # Sort population by fitness (descending)
                population = [x for _, x in sorted(zip(scores, population), reverse=True)]
                scores.sort(reverse=True)

                # Elitism - Retain the best ensembles
                new_population = population[:ELITISM_SIZE]

                # Generate new individuals through crossover and mutation
                while len(new_population) < POPULATION_SIZE:
                    parent1 = tournament_selection(population, scores, TOURNAMENT_SIZE)
                    parent2 = tournament_selection(population, scores, TOURNAMENT_SIZE)
                    child1, child2 = crossover(parent1, parent2)
                    new_population.append(mutate(child1, pool_size))
                    if len(new_population) < POPULATION_SIZE:
                        new_population.append(mutate(child2, pool_size))

                # Update population
                population = new_population

        # Final population sorted by fitness
        final_scores = [calculate_ood_accuracy(ensemble) for ensemble in population]
        best_ensemble = population[final_scores.index(max(final_scores))]
        best_score = max(final_scores)

        return best_ensemble, best_score

    # Run the algorithm
    best_ensemble, best_score = evolutionary_algorithm(pool_size=model_pool.num_classifiers)
    print("Best Ensemble:", best_ensemble)
    print("Best OOD Accuracy Score:", best_score)



