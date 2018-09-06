import numpy as np
import pandas as pd
from algorithms.feature_extraction.feature_extraction import get_extracted_data
from algorithms.feature_extraction.full_database_feature_extraction import save_features_to_csv_file
from algoritmos_felipe.DamageDetection import *
from algorithms.data_imputation.DataImputation import MeanImputation, InterpolationImputation, KNNImputation
from algorithms.constants import RESULTS_DIRECTORY
import json


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
            results['DIs'] = algorithm.DIs
        except Exception as e:
            invalid_iterations += 1
            print(e)

    print('Invalid iterations: {}'.format(invalid_iterations))
    return {key: value / (num_iterations - invalid_iterations) for key, value in results.items()}


# list_missing_data_percentage = [5, 7, 10, 15, 20, 25, 35, 50, 60, 70, 80]
list_missing_data_percentage = [0]

results = {
    'algorithm': [],
    # 'imputation_method': [],
    'missing_data_percentage': [],
    'error_type_I': [],
    'error_type_II': [],
    'true_positives': [],
    'true_negatives': [],
}

# imputation_strategy = KNNImputation
imputation_strategy = None
# imputation_strategies = ['Mean_Imputation', 'Interpolation_Imputation', 'KNN_Imputation_N_3']
imputation_strategies = []
# imputation_strategies = ['Interpolation_Imputation']

dict_DIs = {}

for missing_data_percentage in list_missing_data_percentage:
    num_iterations = 1

    print(">> {}% missing data | {} iterations <<\n".format(missing_data_percentage, num_iterations))

    # save_features_to_csv_file(missing_data_percentage, num_iterations, imputation_strategy)

    algorithms = [
        {'description': 'K-Means',
         'algorithm': K_Means()},
        #
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

    # for imputation_strategy in imputation_strategies:
    for alg in algorithms:
        # print("# {} | {}".format(alg['description'], imputation_strategy))
        print("# {}".format(alg['description']))

        # average_results = get_average_results(alg['algorithm'], missing_data_percentage, num_iterations, imputation_strategy)
        average_results = get_average_results(alg['algorithm'], missing_data_percentage, num_iterations)

        results['algorithm'].append(alg['description'])
        # results['imputation_method'].append(imputation_strategy)
        results['missing_data_percentage'].append(missing_data_percentage)
        results['error_type_I'].append(average_results['error_type_1'])
        results['error_type_II'].append(average_results['error_type_2'])
        results['true_positives'].append(average_results['true_positives'])
        results['true_negatives'].append(average_results['true_negatives'])
        results['true_negatives'].append(average_results['true_negatives'])

        dict_DIs[alg['description']] = average_results['DIs'].tolist()

        print(average_results)
        print()

# plt.legend()
# plt.show()

# df = pd.DataFrame(results)

# print(df)

results_filename = RESULTS_DIRECTORY

with open(results_filename + '/DIs.json', 'w') as fp:
    json.dump(dict_DIs, fp)


# if imputation_strategies:
#     results_filename += '/results_imputation_valendo.csv'
#
# else:
#     results_filename += '/new_results_artigo.csv'
# # results_filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/results/complete_results_amputation.csv'
#
# # df.to_csv(results_filename, mode='a', header=False, index=False)
# df.to_csv(results_filename)
