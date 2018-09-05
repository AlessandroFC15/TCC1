import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import csv
from algorithms.data_imputation.DataImputation import MeanImputation, InterpolationImputation, KNNImputation
from algoritmos_felipe.DamageDetection import *


results_filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation_statistical_measures.csv'

df_imputation = pd.read_csv('/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation/results_imputation_valendo.csv')
df_amputation = pd.read_csv('/home/alessandro/Documentos/Programming/Projects/TCC1/results/complete_results_amputation_2.csv')

# setup the figure and axes
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(111, projection='3d')

# fake data
_x = np.arange(4)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)
width = depth = 1


list_algorithms = [Affinity_Propagation, DBSCAN_Center, Fuzzy_C_Means, G_Means, K_Means]
# list_algorithms = ['K-Means']

# imputation_strategy = "KNN_Imputation_N_7"
imputation_strategies = [MeanImputation, InterpolationImputation, KNNImputation]

list_percentages = [5, 10, 25, 35, 50]

criteria = 'precision'

for percentage in list_percentages:
    results_accuracy = []

    for i, algorithm in enumerate(list_algorithms):
        algorithm_name = algorithm.name
        print(algorithm_name)

        accuracy_values = []
        precision_values = []

        for j, imputation_strategy in enumerate(imputation_strategies):
            individual_results = df_imputation[(df_imputation['algorithm'] == algorithm_name) & (df_imputation['imputation_method'] == imputation_strategy.description) & (df_imputation['missing_data_percentage'] == percentage)]

            precision_list = individual_results['true_positives'] / (individual_results['true_positives'] + individual_results['error_type_I'])
            accuracy_list = (individual_results['true_positives'] + individual_results['true_negatives']) / (
            individual_results['true_positives'] + individual_results['true_negatives'] + individual_results[
                'error_type_I'] + individual_results['error_type_II'])

            accuracy_values.append(accuracy_list.tolist()[0])
            precision_values.append(precision_list.tolist()[0])

        individual_results_amputation = df_amputation[
            (df_amputation['algorithm'] == algorithm_name) & (df_amputation['missing_data_percentage'] == percentage)]

        precision_amp_list = individual_results_amputation['true_positives'] / (individual_results_amputation['true_positives'] + individual_results_amputation['error_type_I'])
        accuracy_amp_list = (individual_results_amputation['true_positives'] + individual_results_amputation[
            'true_negatives']) / (
                                individual_results_amputation['true_positives'] + individual_results_amputation[
                                    'true_negatives'] + individual_results_amputation[
                                    'error_type_I'] + individual_results_amputation['error_type_II'])

        accuracy_values.append(accuracy_amp_list.tolist()[0])
        precision_values.append(precision_amp_list.tolist()[0])

        results_accuracy.append({'algorithm': algorithm, 'accuracy': accuracy_values, 'precision': precision_values})

    with open('{}_{}.csv'.format(criteria, percentage), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Algoritmo'] + [x.descricao for x in imputation_strategies] + ['Amputação'])

        for item in results_accuracy:
            spamwriter.writerow([item['algorithm'].sigla] + [str(100*x).replace('.', ',') for x in item[criteria]])

        # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

# ax1.bar3d(main_x, main_y, main_bottom, width, depth, main_top, shade=True)
# ax1.set_title('Shaded')
#
# plt.show()