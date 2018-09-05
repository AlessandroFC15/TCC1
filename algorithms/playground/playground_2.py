import matplotlib.pyplot as plt
import pandas as pd
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
# missing_data_percentages = [5, 7, 10]
# missing_data_percentages = [70, 80, 90]
colors = ["#0a26af", "#ef6f07", "#1f841b"]

imputation_methods = [MeanImputation, InterpolationImputation, KNNImputation]

fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.99)

results_filename = '/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation_statistical_measures.csv'

df = pd.read_csv(results_filename)

width = 0.35         # the width of the bars

# print(df)

# for missing_data_percentage_imputation in missing_data_percentages:
for i, imputation_method in enumerate(imputation_methods):
    data_imputation = df[df['imputation_method'] == imputation_method.description]

    mse_data = data_imputation['mean_squared_error']

    print(mse_data)
    print('----------')

    p1 = ax.bar(np.array(missing_data_percentages) + (width * i), mse_data, width, color=colors[i])

    # p1

    # print(data_imputation)

    # ax = fig.add_subplot(grid[i, 0])
    #
    # ax.set_xlabel('Dias')
    # ax.set_ylabel('F1 | Frequência (Hz)')
    #
    # ax.plot(range(len(full_data_imputation)), full_data_imputation[:, 1], linewidth=4, label="Imputação por {} | {}".format(imputation_method.descricao, missing_data_percentage_imputation), c=colors[i])
    # ax.plot(range(len(full_data_original)), full_data_original[:, 1], label="Original", c='grey')
    # ax.legend()

ax.set_xticks(missing_data_percentages)
ax.autoscale_view()

plt.show()
