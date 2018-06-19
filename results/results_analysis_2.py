import matplotlib.pyplot as plt
import pandas as pd

df_amputation = pd.read_csv('/home/alessandro/Documentos/Programming/Projects/TCC1/results/new_results.csv')
df_imputation = pd.read_csv(
    '/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation/results_Mean_Imputation.csv')

# list_algorithms = ['K-Means', 'Fuzzy_C_Means', 'Affinity_Propagation', 'GMM', 'G_Means']
list_algorithms = ['K-Means', 'Fuzzy_C_Means', 'Affinity_Propagation', 'GMM', 'G_Means', 'DBSCAN_Center']
# list_algorithms = ['K-Means']

list_colors = ['red', 'green', 'blue', '#f4c141', '#f442df', '#666564']

for i, algorithm_name in enumerate(list_algorithms):
    individual_results = df_imputation[df_imputation['algorithm'] == algorithm_name]
    individual_results_amputation = df_amputation[df_amputation['algorithm'] == algorithm_name]

    sensitivity = individual_results['true_positives'] / (
    individual_results['true_positives'] + individual_results['error_type_II'])
    specificity = individual_results['true_negatives'] / (
    individual_results['true_negatives'] + individual_results['error_type_I'])
    precision = individual_results['true_positives'] / (
    individual_results['true_positives'] + individual_results['error_type_I'])
    recall = sensitivity
    accuracy = (individual_results['true_positives'] + individual_results['true_negatives']) / (
    individual_results['true_positives'] + individual_results['true_negatives'] + individual_results['error_type_I'] +
    individual_results['error_type_II'])

    plt.plot(individual_results['missing_data_percentage'], precision, label=algorithm_name + " (Imputação | Média)",
             linestyle='dashed', color=list_colors[i])

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

    plt.plot(individual_results_amputation['missing_data_percentage'], precision_amputation,
             label=algorithm_name + " (Amputação)", color=list_colors[i])

    plt.ylabel("Precision")
    plt.xlabel("Porcentagem de dados faltantes")
    plt.legend(bbox_to_anchor=(0.1, 0.2), loc=2, borderaxespad=0.)

    plt.xticks([0, 5, 7, 10, 15])

    plt.show()
