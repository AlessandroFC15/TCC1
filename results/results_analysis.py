import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv('/home/alessandro/Documentos/Programming/Projects/TCC1/results/new_results.csv')
df = pd.read_csv('/home/alessandro/Documentos/Programming/Projects/TCC1/results/imputation/results_Mean_Imputation.csv')

list_algorithms = ['K-Means', 'Fuzzy_C_Means', 'DBSCAN_Center', 'Affinity_Propagation', 'GMM', 'G_Means']

for algorithm_name in list_algorithms:
    individual_results = df[df['algorithm'] == algorithm_name]

    print(individual_results)

    plt.plot(individual_results['missing_data_percentage'], individual_results['error_type_I'], label="Erros Tipo 1")
    plt.plot(individual_results['missing_data_percentage'], individual_results['error_type_II'], label="Erros Tipo 2")

    plt.title(algorithm_name)
    plt.ylabel("Número de erros")
    plt.xlabel("Porcentagem de dados faltantes")
    plt.legend(bbox_to_anchor=(0.8, 1.14), loc=2, borderaxespad=0.)

    locs, labels = plt.xticks()

    plt.xticks(individual_results['missing_data_percentage'])

    # plt.savefig('/home/alessandro/Documentos/Programming/Projects/TCC1/results/atualizados/{}.png'.format(algorithm_name))
    plt.show()

    plt.clf()
    plt.cla()
    plt.close()
