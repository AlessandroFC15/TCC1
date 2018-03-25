from algorithms.feature_extraction.full_database_feature_extraction import save_features_to_csv_file
from algoritmos_felipe.DamageDetection import *


def get_average_results(algorithm, missing_data_percentage, num_iterations):
    results = {
        'UCL': 0,
        'error_type_1': 0,
        'error_type_2': 0,
    }

    for i in range(num_iterations):
        all_data = get_extracted_data(missing_data_percentage, i)

        learn_data = all_data[1:158, :]

        algorithm.train_and_test(learn_data, all_data, 197)

        results['UCL'] += algorithm.UCL
        results['error_type_1'] += algorithm.err[0]
        results['error_type_2'] += algorithm.err[1]

    return {key: value / num_iterations for key, value in results.items()}


def get_extracted_data(missing_data_percentage, iteration_number):
    current_file = '/home/alessandro/Documentos/Programming/Projects/TCC1/algorithms/features/Features_Originais_Hora_12_Sensor_5_MDP_{}_{}.csv'.format(
        missing_data_percentage, iteration_number)
    data = genfromtxt(current_file, delimiter=',')

    # Pegando apenas as frequências
    data = data[:, [0, 1]]
    return data


missing_data_percentage = 5
num_iterations = 1

print(">> {}% missing data | {} iterations <<\n".format(missing_data_percentage, num_iterations))

save_features_to_csv_file(missing_data_percentage, num_iterations)

list_num_clusters = range(2, 6)

k_means_algorithms = [{'description': 'KMeans | {} clusters'.format(num_clusters), 'algorithm': K_Means(num_clusters)}
                      for num_clusters in list_num_clusters]
fuzzy_c_means_algorithms = [
    {'description': 'Fuzzy_C_Means | {} clusters'.format(num_clusters), 'algorithm': Fuzzy_C_Means(num_clusters)} for
    num_clusters in list_num_clusters]

algorithms = k_means_algorithms + fuzzy_c_means_algorithms + \
             [
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
    print(get_average_results(alg['algorithm'], missing_data_percentage, num_iterations))
    print()
