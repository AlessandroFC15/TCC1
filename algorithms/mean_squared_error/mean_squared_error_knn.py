import pandas as pd
import os
import re
import math
import time
from algorithms.feature_extraction.feature_extraction import normalize_var, simulate_missing_data
from algoritmos_felipe.DamageDetection import *
from algorithms.data_imputation.DataImputation import MeanImputation, KNNImputation, InterpolationImputation
import scipy


def get_average_results(algorithm, missing_data_percentage, num_iterations, imputation_method=None):
    results = {
        'UCL': 0,
        'error_type_1': 0,
        'error_type_2': 0,
        'true_positives': 0,
        'true_negatives': 0,
    }

    invalid_iterations = 0

    for i in range(num_iterations):
        all_data = get_extracted_data(missing_data_percentage, i if missing_data_percentage != 0 else 0,
                                      imputation_method)

        learn_data = all_data[0:158, :]

        try:
            algorithm.train_and_test(learn_data, all_data, 197)

            results['UCL'] += algorithm.UCL
            results['error_type_1'] += algorithm.err[0]
            results['error_type_2'] += algorithm.err[1]
            results['true_positives'] += len([x for x in algorithm.class_states if x == 1])
            results['true_negatives'] += len([x for x in algorithm.class_states if x == 2])
        except Exception as e:
            invalid_iterations += 1
            print(e)

    print('Invalid iterations: {}'.format(invalid_iterations))
    return {key: value / (num_iterations - invalid_iterations) for key, value in results.items()}


def get_extracted_data(missing_data_percentage, iteration_number, imputation_method=None):
    if imputation_method:
        filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/algorithms/features/{}/' \
                   'Features_Originais_Hora_12_Sensor_5_MDP_{}_{}.csv'.format(imputation_method,
                                                                              missing_data_percentage, iteration_number)
    else:
        filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/algorithms/features/' \
                   'Features_Originais_Hora_12_Sensor_5_MDP_{}_{}.csv'.format(missing_data_percentage, iteration_number)

    data = np.genfromtxt(filename, delimiter=',')

    # Pegando apenas as frequências
    data = data[:, [0, 1]]
    return data


def get_results_imputation_knn(missing_data_percentage, imputation_method):
    canal = 1

    files_directory = '/home/alessandro/FELIPE/Z24_bridge/Z24Date/'

    # Pegar todos os arquivos registrados às 12h
    files = [f for f in os.listdir(files_directory) if re.match(r'.*_12\.mat', f)]
    files.sort()

    for i in range(num_iterations):
        full_database_with_missing_data = []
        full_database = []

        for j, filename in enumerate(files):
            print('#{} | {} | Extracting {} ...'.format(i + 1, j + 1, filename))

            temp = scipy.io.loadmat(files_directory + filename)

            # Obtém apenas a matriz desta observação com os 8 canais.
            dataset_orig = temp['data']

            # Obtém as leituras do canal de interesse.
            dataset_orig = dataset_orig[:, canal]

            # Normalização. (Let's hope this won't break the script).
            dataset_orig = normalize_var(dataset_orig, -1, 1)

            full_database.append(dataset_orig)

            dataset_missing_data = simulate_missing_data(dataset_orig, missing_data_percentage)

            full_database_with_missing_data.append(dataset_missing_data)

        full_database_data_imputed = KNNImputation.impute_data(full_database_with_missing_data)

        results = {
            'sum_squared_error': 0,
            'number_missing_values': 0
        }

        for row_i, row_missing_data in enumerate(full_database_with_missing_data):
            for col_i, col_missing_data in enumerate(row_missing_data):
                if np.isnan(col_missing_data):
                    imputed_value = full_database_data_imputed[row_i][col_i]
                    squared_error = pow((imputed_value - full_database[row_i][col_i]), 2)

                    results['sum_squared_error'] += squared_error
                    results['number_missing_values'] += 1

    mean_squared_error = results['sum_squared_error'] / results['number_missing_values']

    return {
        'mean_squared_error': mean_squared_error,
        'root_mean_squared_error': math.sqrt(mean_squared_error),
    }


list_missing_data_percentage = [60, 70, 80, 90]
# list_missing_data_percentage = [5]

results = {
    'imputation_method': [],
    'missing_data_percentage': [],
    'mean_squared_error': [],
    'root_mean_squared_error': [],
}

# imputation_strategy = KNNImputation
# imputation_strategies = ['Mean_Imputation', 'Interpolation_Imputation', 'KNN_Imputation_N_3', 'KNN_Imputation_N_7']
# imputation_strategies = [MeanImputation, InterpolationImputation, KNNImputation]
imputation_strategies = [KNNImputation]

for missing_data_percentage in list_missing_data_percentage:
    num_iterations = 10

    for imputation_strategy in imputation_strategies:
        print(">> {}% missing data | {} iterations | {} <<\n".format(missing_data_percentage, num_iterations, imputation_strategy.description))

        results_estimation = get_results_imputation_knn(missing_data_percentage, imputation_strategy)

        results['imputation_method'].append(imputation_strategy.description)
        results['missing_data_percentage'].append(missing_data_percentage)
        results['mean_squared_error'].append(results_estimation['mean_squared_error'])
        results['root_mean_squared_error'].append(results_estimation['root_mean_squared_error'])

df = pd.DataFrame(results)

print(df)

results_filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation_statistical_measures.csv'

df.to_csv(results_filename, mode='a', header=False, index=False)
