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
        "precision": (num_acertos / len(resultado_final)) * 100,
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

