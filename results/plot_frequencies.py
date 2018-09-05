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
missing_data_list = [25]

for missing_data_percentage_imputation in missing_data_list:
    imputation_method = KNNImputation
    iteration_number = 0

    dot_size = 20

    full_data_0_percent = get_extracted_data(missing_data_percentage=0, iteration_number=0)
    full_data_imputation = get_extracted_data(missing_data_percentage_imputation, iteration_number, imputation_method)
    full_data_amputation = get_extracted_data(missing_data_percentage_imputation, iteration_number=0)

    plt.xlabel('Número da observação')
    plt.ylabel('Valor medido pelo sensor 5')
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.99)
    plt.scatter(range(len(full_data_0_percent)), full_data_0_percent[:, 0],  s=dot_size, label="Valor Inalterado")
    plt.scatter(range(len(full_data_imputation)), full_data_imputation[:, 0],  s=dot_size, label="Valor Faltante")

    # plt.plot(range(len(full_data_0_percent)), full_data_0_percent[:, 1], label="F3")
    # plt.scatter(range(len(full_data_imputation)), full_data_imputation[:, 0], s=20, cmap='viridis', label="{}% | Imputação por Média".format(missing_data_percentage_imputation))
    # plt.scatter(range(len(full_data_amputation)), full_data_amputation[:, 0], s=dot_size, cmap='viridis', label="{}%".format(missing_data_percentage_imputation))

    # plt.scatter(full_data_0_percent[:, 0], full_data_0_percent[:, 1], s=20, cmap='viridis', label="Frequência original".format(0))
    # plt.scatter(full_data_amputation[:, 0], full_data_amputation[:, 1], s=20, cmap='viridis', label="Frequência com {}% de MV".format(missing_data_percentage_imputation))
    plt.legend()
    # plt.savefig(
    #     '/home/alessandro/Documentos/Programming/Projects/TCC1/graphics/imagens_TCC/f1_vs_f3_0_{}.png'.format(missing_data_percentage_imputation))
    plt.show()

    plt.clf()



    # plt.xlabel('Dias')
    # plt.ylabel('F2 | Frequência (Hz)')
    # plt.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.99)
    #
    # plt.scatter(range(len(full_data_0_percent)), full_data_0_percent[:, 1],  c='#b50707', s=dot_size, label="F2 original")
    # plt.scatter(range(len(full_data_amputation)), full_data_amputation[:, 1], c='#31871c', s=dot_size, cmap='viridis', label="F2 após amputação de {}% dos dados".format(missing_data_percentage_imputation))
    #
    # plt.legend()
    #
    # plt.savefig(
    #     '/home/alessandro/Documentos/Programming/Projects/TCC1/graphics/imagens_TCC/f2_0_{}.png'.format(
    #         missing_data_percentage_imputation))
    # plt.show()