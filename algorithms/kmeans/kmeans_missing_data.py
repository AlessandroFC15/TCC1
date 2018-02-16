import scipy.io
import numpy
import matplotlib.pyplot as plt
import sys
import random
from algorithms.kmeans.kmeans import get_kmeans_results


def get_database_with_missing_data(data, missing_data_percentage):
    num_samples_undamaged_data = 3470

    num_samples_to_be_removed = round(len(data) * (missing_data_percentage / 100))

    new_database = list(data)

    for i in range(0, num_samples_to_be_removed):
        rand_num = random.randint(0, len(new_database) - 1)

        if rand_num < num_samples_undamaged_data:
            num_samples_undamaged_data -= 1

        del new_database[rand_num]

    return {
        'database': numpy.asarray(new_database),
        'num_samples_undamaged_data': num_samples_undamaged_data
    }


def get_kmeans_precision(database, num_iterations, missing_data_percentage):
    precision_total = 0

    for i in range(0, num_iterations):
        new_database = get_database_with_missing_data(database, missing_data_percentage)

        data = new_database['database']

        n = len(data)
        nc = new_database['num_samples_undamaged_data']
        nb = round(0.9 * nc)

        # Amostras de 1-nb correspondem a dados sem dano usados para treino
        undamaged_data_train = data[:nb]

        print(len(undamaged_data_train))

        total_undamaged_data = data[:nc]

        # As amostras no intervalo nc-n são as amostras com dano usadas na fase de teste.
        damaged_data = data[nc:n]

        score_labels = [1 for data in database[:3470]] + [0 for y in database[3470:3932]]

        # optimal_K = find_optimal_num_clusters(undamaged_data_train, data, score_labels)
        optimal_K = 3

        kmeans_results = get_kmeans_results(optimal_K, undamaged_data_train, database, score_labels)

        precision_total += kmeans_results['precision']
        print('Total amostras: {}'.format(len(data)))

        if i == num_iterations - 1:
            # plt.scatter(range(0, len(data)), data[:, 3], c=kmeans_results['final_result'], s=5, cmap='viridis')
            plt.scatter(range(0, 3932), database[:, 3], c=kmeans_results['final_result'], s=5, cmap='viridis')

    return precision_total / num_iterations


mat = scipy.io.loadmat('../../database/databasez24.mat')

data = mat['databasez24'][0][0][0]

num_iterations = 100
missing_data_percentage = 99.5

print(get_kmeans_precision(data, num_iterations, missing_data_percentage))

plt.show()

