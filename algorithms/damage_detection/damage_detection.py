import numpy as np
import pandas as pd

from algoritmos_felipe.DamageDetection import *


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
                   'Features_Originais_Hora_12_Sensor_5_MDP_{}_{}.csv'.format(imputation_method.description,
                                                                              missing_data_percentage, iteration_number)
    else:
        filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/algorithms/features/' \
                   'Features_Originais_Hora_12_Sensor_5_MDP_{}_{}.csv'.format(missing_data_percentage, iteration_number)

    data = np.genfromtxt(filename, delimiter=',')

    # Pegando apenas as frequências
    data = data[:, [0, 1]]
    return data


list_missing_data_percentage = [5, 7, 10, 15]

results = {
    'algorithm': [],
    'missing_data_percentage': [],
    'error_type_I': [],
    'error_type_II': [],
    'true_positives': [],
    'true_negatives': [],
}

# imputation_strategy = DataImputation.MeanImputation
imputation_strategy = None

for missing_data_percentage in list_missing_data_percentage:
    num_iterations = 25

    print(">> {}% missing data | {} iterations <<\n".format(missing_data_percentage, num_iterations))

    # save_features_to_csv_file(missing_data_percentage, num_iterations, imputation_strategy)

    algorithms = [
        {'description': 'K-Means',
         'algorithm': K_Means()},

        {'description': 'Fuzzy_C_Means',
         'algorithm': Fuzzy_C_Means()},

        {'description': 'DBSCAN_Center',
         'algorithm': DBSCAN_Center(0.09, 3)},

        {'description': 'Affinity_Propagation',
         'algorithm': Affinity_Propagation()},

        {'description': 'GMM',
         'algorithm': GMM()},

        {'description': 'G_Means',
         'algorithm': G_Means()},
    ]

    for alg in algorithms:
        print("# " + alg['description'])

        average_results = get_average_results(alg['algorithm'], missing_data_percentage, num_iterations,
                                              imputation_strategy)

        results['algorithm'].append(alg['description'])
        results['missing_data_percentage'].append(missing_data_percentage)
        results['error_type_I'].append(average_results['error_type_1'])
        results['error_type_II'].append(average_results['error_type_2'])
        results['true_positives'].append(average_results['true_positives'])
        results['true_negatives'].append(average_results['true_negatives'])

        print(average_results)
        print()

# plt.legend()
# plt.show()

df = pd.DataFrame(results)

print(df)

if imputation_strategy:
    results_filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation/' \
                       'results_{}.csv'.format(imputation_strategy.description)
else:
    results_filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/results/new_results_3.csv'

df.to_csv(results_filename)
