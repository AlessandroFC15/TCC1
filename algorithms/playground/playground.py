import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
from algorithms.data_imputation.DataImputation import MeanImputation, InterpolationImputation, KNNImputation


def get_extracted_data(missing_data_percentage, iteration_number, imputation_method=None):
    if imputation_method:
        print(imputation_method.description)

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


missing_data_percentages = [5, 7, 10, 15, 20, 25, 35, 50, 60, 70, 80, 90]
# missing_data_percentages = [25]
colors = ["#0a26af", "#ef6f07", "#1f841b"]

imputation_methods = [MeanImputation, InterpolationImputation, KNNImputation]

for missing_data_percentage_imputation in missing_data_percentages:
    grid = plt.GridSpec(3, 1, wspace=0.2, hspace=0.25)
    fig = plt.figure(figsize=(9, 7))

    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.99)

    for i, imputation_method in enumerate(imputation_methods):

        full_data_original = get_extracted_data(0, iteration_number=0)
        full_data_imputation = get_extracted_data(missing_data_percentage_imputation, iteration_number=0, imputation_method=imputation_method)

        ax = fig.add_subplot(grid[i, 0])

        ax.set_xlabel('Dias')
        ax.set_ylabel('F3 | Frequência (Hz)')

        ax.plot(range(len(full_data_imputation)), full_data_imputation[:, 1], linewidth=4, label="Imputação por {} | {}%".format(imputation_method.descricao, missing_data_percentage_imputation), c=colors[i])
        ax.plot(range(len(full_data_original)), full_data_original[:, 1], label="Original", c='grey')
        ax.legend()

    plt.show()
    # fig.savefig(
    #    '/home/alessandro/Documentos/Programming/Projects/TCC1/graphics/apresentacao_resultados/Comparison_F1_F3_Imputation/comparison_f3_{}.png'.format(missing_data_percentage_imputation))

