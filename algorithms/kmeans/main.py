import matplotlib.pyplot as plt

from algorithms.kmeans.kmeans import *

data = get_database()['database']

nb = 3123
nc = 3470
n = 3932

# Amostras de 1-3123 correspondem a dados sem dano usados para treino
undamaged_data_train = data[:nb]

# As amostras de 3124-3470 são amostras sem dano usadas para teste
undamaged_data_test = data[nb:nc]

total_undamaged_data = data[:nc]

# As amostras no intervalo 3471-3932 são as amostras com dano usadas na fase de teste.
damaged_data = data[nc:n]

score_labels = [1 for data in total_undamaged_data] + [0 for y in damaged_data]

print('>> Finding optimal K')
# optimal_K = find_optimal_num_clusters(undamaged_data_train, data, score_labels)
optimal_K = 2

kmeans_results = get_kmeans_results(optimal_K, undamaged_data_train, data, score_labels)

print(kmeans_results['precision'])

# plt.scatter(range(0, n), data[:, 0], c=kmeans_results['final_result'], s=5, cmap='viridis')

plt.show()
