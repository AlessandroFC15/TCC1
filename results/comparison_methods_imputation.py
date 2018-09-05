import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

df_amputation = pd.read_csv('/home/alessandro/Documentos/Programming/Projects/TCC1/results/complete_results_amputation_2.csv')
df_imputation = pd.read_csv(
    '/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation/results_imputation.csv')

# list_algorithms = ['K-Means', 'Fuzzy_C_Means', 'Affinity_Propagation', 'GMM', 'G_Means']
list_algorithms = ['K-Means', 'Fuzzy_C_Means', 'Affinity_Propagation', 'GMM', 'G_Means', 'DBSCAN_Center']
# list_algorithms = ['K-Means']

# imputation_strategy = "KNN_Imputation_N_7"
imputation_strategies = ["Mean_Imputation", "Interpolation_Imputation", "KNN_Imputation_N_3", "KNN_Imputation_N_7"]

list_colors = ['green', 'blue', '#f4c141', '#f442df']

lista_criterios = ["Accuracy", "Sensitivity", "Specificity", "Precision"]
# lista_criterios = ["Sensitivity"]

for criterio_analisado in lista_criterios:
    for i, algorithm_name in enumerate(list_algorithms):
        individual_results_amputation = df_amputation[df_amputation['algorithm'] == algorithm_name].sort_values(by=['missing_data_percentage'])

        sensitivity_amputation = individual_results_amputation['true_positives'] / (
        individual_results_amputation['true_positives'] + individual_results_amputation['error_type_II'])
        specificity_amputation = individual_results_amputation['true_negatives'] / (
        individual_results_amputation['true_negatives'] + individual_results_amputation['error_type_I'])
        precision_amputation = individual_results_amputation['true_positives'] / (
        individual_results_amputation['true_positives'] + individual_results_amputation['error_type_I'])
        accuracy_amputation = (individual_results_amputation['true_positives'] + individual_results_amputation[
            'true_negatives']) / (
                                  individual_results_amputation['true_positives'] + individual_results_amputation[
                                      'true_negatives'] + individual_results_amputation['error_type_I'] +
                                  individual_results_amputation['error_type_II'])

        criterios_amputation = {
            'sensitivity': sensitivity_amputation,
            'specificity': specificity_amputation,
            'precision': precision_amputation,
            'accuracy': accuracy_amputation,
        }

        plt.plot(individual_results_amputation['missing_data_percentage'],
                 criterios_amputation[criterio_analisado.lower()],
                 label="Amputação", color='red')

        for j, imputation_strategy in enumerate(imputation_strategies):
            individual_results = df_imputation[(df_imputation['algorithm'] == algorithm_name) & (df_imputation['imputation_method'] == imputation_strategy)]

            sensitivity = individual_results['true_positives'] / (individual_results['true_positives'] + individual_results['error_type_II'])
            specificity = individual_results['true_negatives'] / (individual_results['true_negatives'] + individual_results['error_type_I'])
            precision = individual_results['true_positives'] / (individual_results['true_positives'] + individual_results['error_type_I'])
            recall = sensitivity
            accuracy = (individual_results['true_positives'] + individual_results['true_negatives']) / (individual_results['true_positives'] + individual_results['true_negatives'] + individual_results['error_type_I'] + individual_results['error_type_II'])

            criterios = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'accuracy': accuracy,
            }

            plt.plot(individual_results['missing_data_percentage'], criterios[criterio_analisado.lower()], label=imputation_strategy,
                     linestyle='dashed', color=list_colors[j])

        plt.ylabel(criterio_analisado)
        plt.xlabel("Porcentagem de dados faltantes")
        # plt.legend(bbox_to_anchor=(0.1, 0.2), loc=2, borderaxespad=0.)

        plt.legend()
        plt.xticks([0, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50])

        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.10, top=0.9)

        plt.title('Comparação entre métodos de imputação | {} | {}'.format(criterio_analisado, algorithm_name))
        plt.savefig('/home/alessandro/Documentos/Programming/Projects/TCC1/graphics/apresentacao_resultados/{}_{}.png'.format(criterio_analisado, algorithm_name))

        # plt.show()
        plt.clf()

