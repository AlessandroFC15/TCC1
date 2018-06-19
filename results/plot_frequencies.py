import matplotlib.pyplot as plt
import numpy as np


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


full_data_0_percent = get_extracted_data(missing_data_percentage=0, iteration_number=0)
full_data_5_percent = get_extracted_data(missing_data_percentage=10, iteration_number=0)

plt.xlabel('f1')
plt.ylabel('f2')
plt.scatter(full_data_0_percent[:, 0], full_data_0_percent[:, 1], s=20, cmap='viridis', label="{}%".format(0))
plt.scatter(full_data_5_percent[:, 0], full_data_5_percent[:, 1], s=20, cmap='viridis', label="{}%".format(10))

plt.legend()
plt.show()
