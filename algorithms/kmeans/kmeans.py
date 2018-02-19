import matplotlib.pyplot as plt
import numpy
import random
import scipy.io
from sklearn import metrics
from sklearn.cluster import KMeans


def get_UCL(kmeans_model, X):
    resultado = kmeans_model.transform(X)

    vetor_y = [min(x) for x in resultado]

    vetor_y.sort()

    Y = numpy.asarray(vetor_y)

    return Y[int(numpy.rint(Y.size * 0.95))]


def plot_clusters(kmeans_model, train_data):
    labels = kmeans_model.labels_

    plt.scatter(range(0, len(train_data)), train_data[:, 0], c=labels, s=5, cmap='viridis')
    # plt.scatter(range(0, len(train_data)), train_data[:, 1], c=labels, s=5, cmap='viridis')


def get_kmeans_results(num_clusters, train_data, test_data, test_labels):
    kmeans_model = KMeans(n_clusters=num_clusters).fit(train_data)
    labels = kmeans_model.labels_

    # plot_clusters(kmeans_model, train_data)

    UCL = get_UCL(kmeans_model, train_data)

    ### Teste na Base toda

    resultado_teste = kmeans_model.transform(test_data)

    vetor_y_teste = [min(x) for x in resultado_teste]

    vetor_y_teste.sort()

    resultado_final = [(0 if dado > UCL else 1) for dado in vetor_y_teste]

    num_acertos = 0

    for i, dado in enumerate(resultado_final):
        if dado == test_labels[i]:
            num_acertos += 1

    return {
        "precision": (num_acertos / len(resultado_final)) * 100,
        "final_result": resultado_final,
        "kmeans_model": kmeans_model,
        "calinski_score": metrics.calinski_harabaz_score(train_data, labels),
        "silhouette_score": metrics.silhouette_score(train_data, labels, metric='euclidean')
    }


def find_optimal_num_clusters(train_data, test_data, test_labels):
    optimal = {
        'K': 0,
        'score': -1
    }

    for k in range(2, 21):
        kmeans_precision = get_kmeans_results(k, train_data, test_data, test_labels)["precision"]

        if kmeans_precision > optimal['score']:
            optimal['K'] = k
            optimal['score'] = kmeans_precision

    return optimal['K']


def get_database(missing_data_percentage=0):
    mat = scipy.io.loadmat('../../database/databasez24.mat')

    data = mat['databasez24'][0][0][0]

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


def get_kmeans_precision(num_iterations, missing_data_percentage=0, num_clusters=3):
    database = get_database()['database']

    precision_total = 0
    calinski_total = 0
    silhouette_total = 0

    for i in range(0, num_iterations):
        new_database = get_database(missing_data_percentage)

        data = new_database['database']

        n = len(data)
        nc = new_database['num_samples_undamaged_data']
        nb = round(0.9 * nc)

        # Amostras de 1-nb correspondem a dados sem dano usados para treino
        undamaged_data_train = data[:nb]

        # print(len(undamaged_data_train))

        total_undamaged_data = data[:nc]

        # As amostras no intervalo nc-n são as amostras com dano usadas na fase de teste.
        damaged_data = data[nc:n]

        score_labels = [1 for data in database[:3470]] + [0 for y in database[3470:3932]]

        # optimal_K = find_optimal_num_clusters(undamaged_data_train, data, score_labels)
        optimal_K = num_clusters

        kmeans_results = get_kmeans_results(optimal_K, undamaged_data_train, database, score_labels)

        precision_total += kmeans_results['precision']
        calinski_total += kmeans_results['calinski_score']
        silhouette_total += kmeans_results['silhouette_score']

        if i == num_iterations - 1:
            print('Total amostras: {}'.format(len(data)))

            # plt.scatter(range(0, len(data)), data[:, 3], c=kmeans_results['final_result'], s=5, cmap='viridis')
            # plt.scatter(range(0, 3932), database[:, 0], c=kmeans_results['final_result'], s=5, cmap='viridis')

    print(calinski_total / num_iterations)
    print(silhouette_total / num_iterations)

    return precision_total / num_iterations
