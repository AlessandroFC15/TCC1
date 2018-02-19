import matplotlib.pyplot as plt

from algorithms.kmeans.kmeans import get_kmeans_precision

num_iterations = 100
missing_data_percentage = 99

optimal_K = {
    "k": -1,
    "precision": -1
    }

for k in range(2, 7):
    print(k)

    precision = get_kmeans_precision(num_iterations, missing_data_percentage, k)
    print(precision)

    if precision > optimal_K['precision']:
        optimal_K['k'] = k
        optimal_K['precision'] = precision

    print('-----------------')

print(optimal_K)

plt.show()

# plt.show()
