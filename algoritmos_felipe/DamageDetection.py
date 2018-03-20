from __future__ import division, print_function

import numpy as np
import scipy.spatial.distance as distances
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.mixture import GaussianMixture

from algoritmos_felipe.GMeans import GMeans


class DummySubplot:
    def plot(self, *args):
        pass

    def add_artist(self, *args):
        pass

    def axis(self, *args):
        pass


class DamageDetection:
    def __init__(self):
        self.DIs = None
        self._train_DIs = None
        self._data_break_point = None
        pass

    def train_and_test(self, training_data, test_data, break_point):
        pass

    def _set_resulting_parameters(self):
        # CALCULAR THRESHOLD
        lv = 0.95
        n = len(self.DIs)

        UCL = self._train_DIs[int(np.floor(len(self._train_DIs) * lv))]
        flag = np.zeros(n, dtype=bool)
        flag[self._data_break_point:] = 1

        # CALCULAR ACERTOS E ERROS TIPO I e II
        class_states = np.zeros(n)
        err_t1 = 0
        err_t2 = 0
        for i in range(n):
            if self.DIs[i] > UCL:
                if flag[i] == 0:
                    class_states[i] = 3
                    err_t1 += 1
                else:
                    class_states[i] = 1
            elif flag[i] == 1:
                class_states[i] = 4
                err_t2 += 1
            else:
                class_states[i] = 2

        # CLASS STATE DIZ 1 = VERDADEIRO POSITIVO, 2=VERDADEIRO NEGATIVO, 3=FALSO POSITIVO, 4=FALSO NEGATIVO
        self.class_states = class_states
        self.UCL = UCL
        self.err = (err_t1, err_t2)


# FUNCIONANDO_TESTE/LIMPO/
class K_Means(DamageDetection):
    def __init__(self, number_of_clusters):
        super().__init__()
        self.number_of_clusters = number_of_clusters

    def train_and_test(self, training_data, test_data, break_point):
        # Aplica o KMEANS na base de dados de treino
        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0).fit(training_data)

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TREINO
        train_dist = kmeans.transform(training_data)
        train_min_dist = np.zeros(len(training_data))
        for i in range(len(train_dist)):
            train_min_dist[i] = min(train_dist[i])
        train_min_dist.sort()

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TESTE
        test_dist = kmeans.transform(test_data)
        test_min_dist = np.zeros(len(test_dist))
        for i in range(len(test_dist)):
            test_min_dist[i] = min(test_dist[i])

        self.DIs = test_min_dist
        self._train_DIs = train_min_dist
        self._data_break_point = break_point

        self._set_resulting_parameters()


# FUNCIONANDO_TESTE/LIMPO
class Fuzzy_C_Means(DamageDetection):
    def __init__(self, number_of_clusters):
        super().__init__()
        self.number_of_clusters = number_of_clusters

    def train_and_test(self, training_data, test_data, break_point):
        cntr, u, _, d, _, _, fpc = fuzz.cluster.cmeans(
            training_data.T, self.number_of_clusters, 5, error=0.005, maxiter=1000, init=None, seed=1)

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TREINO
        train_min_dist = np.min(d, axis=0)
        train_min_dist.sort()

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TESTE
        _, _, d, _, _, _ = fuzz.cluster.cmeans_predict(
            test_data.T, cntr, 2, error=0.005, maxiter=1000)

        test_min_dist = np.min(d, axis=0)

        self.DIs = test_min_dist
        self._train_DIs = train_min_dist
        self._data_break_point = break_point

        self._set_resulting_parameters()


# FUNCIONANDO_TESTE/LIMPO
class DBSCAN_Center(DamageDetection):
    def __init__(self, eps, min_points):
        super().__init__()
        self.eps = eps
        self.min_points = min_points

    def train_and_test(self, training_data, test_data, break_point):
        db = DBSCAN(self.eps, self.min_points).fit(training_data)

        # CALCULAR CENTROS

        labels = np.unique(db.labels_)
        max_label = max(labels)
        centers = np.zeros((max_label + 1, 2))
        for i in range(max_label + 1):
            if max_label == 0 and len(labels) == 1:
                indices = np.argwhere(db.labels_ == labels[i]).T[0]
            else:
                indices = np.argwhere(db.labels_ == labels[i + 1]).T[0]
            points = training_data[indices]
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            centroid = (sum(x) / len(points), sum(y) / len(points))
            centers[i] = centroid

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TREINO

        train_min_dist = np.zeros(len(training_data))
        for i, point in enumerate(training_data):
            if db.labels_[i] == -1:
                train_min_dist[i] = -1
                continue
            train_min_dist[i] = distances.euclidean(point, centers[db.labels_[i]])

        train_min_dist = train_min_dist[train_min_dist >= 0]
        train_min_dist.sort()

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TESTE

        n = len(test_data)
        test_min_dist = np.zeros((n, max_label + 1))
        for i, point in enumerate(test_data):
            for j, center in enumerate(centers):
                test_min_dist[i, j] = distances.euclidean(point, center)

        test_min_dist = np.min(test_min_dist, axis=1)

        self.DIs = test_min_dist
        self._train_DIs = train_min_dist
        self._data_break_point = break_point

        self._set_resulting_parameters()


# FUNCIONANDO_TESTE/LIMPO
class Affinity_Propagation(DamageDetection):
    def __init__(self):
        super().__init__()

    def train_and_test(self, training_data, test_data, break_point):
        aff = AffinityPropagation().fit(training_data)

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TREINO
        train_min_dist = np.zeros(len(training_data))
        for i, point in enumerate(training_data):
            if aff.labels_[i] == -1:
                train_min_dist[i] = -1
                continue
            train_min_dist[i] = distances.euclidean(point, aff.cluster_centers_[aff.labels_[i]])

        train_min_dist = train_min_dist[train_min_dist >= 0]
        train_min_dist.sort()

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TESTE
        n = len(test_data)

        test_min_dist = np.zeros((n, len(aff.cluster_centers_)))
        for i, point in enumerate(test_data):
            for j, center in enumerate(aff.cluster_centers_):
                test_min_dist[i, j] = distances.euclidean(point, center)

        test_min_dist = np.min(test_min_dist, axis=1)

        self.DIs = test_min_dist
        self._train_DIs = train_min_dist
        self._data_break_point = break_point

        self._set_resulting_parameters()


# FUNCIONANDO_TESTE/LIMPO
class GMM(DamageDetection):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def train_and_test(self, training_data, test_data, break_point):
        n_components = self.n_components

        gmm = GaussianMixture(n_components=n_components).fit(training_data)

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TREINO
        train_min_dist = np.zeros(len(training_data))
        train_predicted = gmm.predict(training_data)
        for i, point in enumerate(training_data):
            train_min_dist[i] = distances.euclidean(point, gmm.means_[train_predicted[i]])

        train_min_dist = train_min_dist[train_min_dist >= 0]
        train_min_dist.sort()

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TESTE
        n = len(test_data)

        test_min_dist = np.zeros((n, n_components))
        for i, point in enumerate(test_data):
            for j, center in enumerate(gmm.means_):
                test_min_dist[i, j] = distances.euclidean(point, center)
        test_min_dist = np.min(test_min_dist, axis=1)

        self.DIs = test_min_dist
        self._train_DIs = train_min_dist
        self._data_break_point = break_point

        self._set_resulting_parameters()


class G_Means(DamageDetection):
    def __init__(self):
        super().__init__()

    def train_and_test(self, training_data, test_data, break_point):
        gmeans = GMeans(strictness=4)
        gmeans.fit(training_data)

        # CALCULAR CENTROS
        labels = np.unique(gmeans.labels_)
        max_label = len(labels)
        centers = np.zeros((max_label + 1, training_data.shape[1]))
        a = {}
        for i, label in enumerate(labels):
            indices = np.argwhere(gmeans.labels_ == label).T[0]
            points = training_data[indices]
            a.update({label: i})
            for j in range(training_data.shape[1]):
                centers[i, j] = sum([p[j] for p in points]) / len(points)

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TREINO
        train_min_dist = np.zeros(len(training_data))
        for i, point in enumerate(training_data):
            train_min_dist[i] = distances.euclidean(point, centers[a[gmeans.labels_[i]]])

        train_min_dist = train_min_dist[train_min_dist >= 0]
        train_min_dist.sort()

        # ACHAR MENORES DISTANCIAS PARA OS DADOS DE TESTE

        n = len(test_data)
        test_min_dist = np.zeros((n, max_label + 1))
        for i, point in enumerate(test_data):
            for j, center in enumerate(centers):
                test_min_dist[i, j] = distances.euclidean(point, center)

        test_min_dist = np.min(test_min_dist, axis=1)

        self.DIs = test_min_dist
        self._train_DIs = train_min_dist
        self._data_break_point = break_point
        self._number_of_clusters = max_label
        self._set_resulting_parameters()


def method_name_1():
    current_file = 'C:/Users/Felipe/Documents/UFPA/TCC/manoel_afonso-matlab-0d7ca42aed75' \
                   '/sinal_aproximado_normalizado/original/Features_Orig_Hora_12_COL_2.csv'
    data = np.genfromtxt(current_file, delimiter=',')
    data = data[:, [0, 1]]
    return data


if __name__ == "__main__":
    from numpy import genfromtxt

    all_data = method_name_1()

    print(all_data.shape)
    learn_data = all_data[1:158, :]

    alg = DBSCAN_Center(0.09, 3)
    alg.train_and_test(learn_data, all_data, 197)
    print(alg.err)

    # for i in range(158):
    #    print(euclidean_distances(learn_data[0,:],learn_data[1,:]))


    '''alg.train_and_test(learn_data, all_data, 197)

    dis_1 = alg.DIs
    UCL_1 = alg.UCL
    print(alg.err)

    plt.plot(range(234), dis_1, 'o')
    plt.axhline(y=UCL_1, color='k', linestyle='-')

    plt.axis('tight')

    plt.show()'''
