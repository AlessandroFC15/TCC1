import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
from algorithms.data_imputation.DataImputation import MeanImputation, InterpolationImputation, KNNImputation


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

rcParams['figure.figsize'] = 9, 4

# missing_data_list = [5, 7, 10, 15, 20, 50, 70, 90]
missing_data_list = [5, 7, 10]

for missing_data_percentage_imputation in missing_data_list:
    grid = plt.GridSpec(2, 1, wspace=0.2, hspace=0.2)
    fig = plt.figure(figsize=(9, 7))

    imputation_method = KNNImputation
    iteration_number = 0

    dot_size = 20

    full_data_0_percent = get_extracted_data(missing_data_percentage=0, iteration_number=0)
    # full_data_imputation = get_extracted_data(missing_data_percentage_imputation, iteration_number, imputation_method)
    full_data_amputation = get_extracted_data(missing_data_percentage_imputation, iteration_number=0)

    fi1_ax = fig.add_subplot(grid[0, 0])
    fi2_ax = fig.add_subplot(grid[1, 0])

    fi1_ax.set_xlabel('Dias')
    fi1_ax.set_ylabel('F1 | Frequência (Hz)')
    # plt.ylabel()
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.99)
    fi1_ax.scatter(range(len(full_data_0_percent)), full_data_0_percent[:, 0],  s=dot_size, label="F1 original")

    # plt.plot(range(len(full_data_0_percent)), full_data_0_percent[:, 1], label="F3")
    # plt.scatter(range(len(full_data_imputation)), full_data_imputation[:, 0], s=20, cmap='viridis', label="{}% | Imputação por Média".format(missing_data_percentage_imputation))
    fi1_ax.scatter(range(len(full_data_amputation)), full_data_amputation[:, 0], s=dot_size, cmap='viridis', label="F1 após amputação de {}% dos dados".format(missing_data_percentage_imputation))

    fi1_ax.legend()

    # plt.scatter(full_data_5_percent[:, 0], full_data_5_percent[:, 1], s=20, cmap='viridis', label="{}%".format(10))
    # plt.scatter(full_data_5_percent[:, 0], full_data_5_percent[:, 1], s=20, cmap='viridis', label="{}%".format(10))
    # plt.legend()
    # plt.savefig(
    #     '/home/alessandro/Documentos/Programming/Projects/TCC1/graphics/imagens_TCC/f1_0_{}.png'.format(missing_data_percentage_imputation))
    # plt.show()

    # plt.clf()

    fi2_ax.set_xlabel('Dias')
    fi2_ax.set_ylabel('F3 | Frequência (Hz)')
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.06, top=0.99)

    fi2_ax.scatter(range(len(full_data_0_percent)), full_data_0_percent[:, 1],  c='#b50707', s=dot_size, label="F3 original")
    fi2_ax.scatter(range(len(full_data_amputation)), full_data_amputation[:, 1], c='#31871c', s=dot_size, cmap='viridis', label="F3 após amputação de {}% dos dados".format(missing_data_percentage_imputation))

    fi2_ax.legend()

    plt.savefig(
        '/home/alessandro/Documentos/Programming/Projects/TCC1/graphics/imagens_TCC/f1_f3_0_{}.png'.format(
            missing_data_percentage_imputation))
    plt.show()