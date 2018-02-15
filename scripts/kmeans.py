import scipy.io
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import sys


def get_UCL(kmeans_model, X):
    resultado = kmeans_model.transform(X)

    vetor_y = [min(x) for x in resultado]

    vetor_y.sort()

    Y = numpy.asarray(vetor_y)

    return Y[int(numpy.rint(Y.size * 0.95))]


def get_kmeans_results(num_clusters, train_data, test_data, test_labels):
    kmeans_model = KMeans(n_clusters=num_clusters).fit(train_data)

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
        "precision": (num_acertos / n) * 100,
        "final_result": resultado_final,
        "kmeans_model": kmeans_model
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


mat = scipy.io.loadmat('../database/databasez24.mat')

data = mat['databasez24'][0][0][0]

nb = 3123
nc = 3470
n = 3932
m = 4
lv = 0.95

# Amostras de 1-3123 correspondem a dados sem dano usados para treino
undamaged_data_train = data[:nb]

# As amostras de 3124-3470 são amostras sem dano usadas para teste
undamaged_data_test = data[nb:nc]

total_undamaged_data = data[:nc]

# As amostras no intervalo 3471-3932 são as amostras com dano usadas na fase de teste.
damaged_data = data[nc:n]

score_labels = [1 for data in total_undamaged_data] + [0 for y in damaged_data]

print('>> Finding optimal K')
optimal_K = find_optimal_num_clusters(undamaged_data_train, data, score_labels)

kmeans_results = get_kmeans_results(optimal_K, undamaged_data_train, data, score_labels)

plt.scatter(range(0, n), data[:, 1], c=kmeans_results['final_result'], s=5, cmap='viridis')

plt.show()
