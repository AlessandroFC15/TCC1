import numpy as np
import random
import scipy.io
import matplotlib.pyplot as plt
import os
import re
from algorithms.feature_extraction.step_extraction import step_extraction
from algorithms.data_imputation.DataImputation import KNNImputation


def simulate_missing_data(data, missing_data_percentage):
    num_samples_to_be_removed = round(len(data) * (missing_data_percentage / 100))

    new_database = list(data)

    for i in range(0, num_samples_to_be_removed):
        while True:
            rand_num = random.randint(0, len(new_database) - 1)

            if not np.isnan(new_database[rand_num]):
                new_database[rand_num] = np.NaN
                break

    return new_database


def realizar_amputacao_dados(data):
    return [x for x in data if not np.isnan(x)]


def normalize_var(array, x, y):
    """
    Normaliza os valores de um array no intervalo [x,y].

    :param array: Valores originais.
    :param x: Intervalo para a normalização (ex.: [0,1]).
    :param y: Intervalo para a normalização (ex.: [0,1]).
    :return: Valores normalizados
    """

    m = min(array)

    range = max(array) - m

    array = (array - m) / range

    range2 = y - x
    normalized = (array * range2) + x

    return normalized


def plot_comparison_imputed_values_with_missing_values(original_dataset, dataset_missing_data, dataset_imputed_values):
    dot_size = 15

    plt.scatter(range(0, len(original_dataset)), original_dataset, s=dot_size, color='#26539b', label='Valor Inalterado')

    data_plot_missing_values = [{'position': i, 'value': original_dataset[i]} for i, value in enumerate(dataset_missing_data) if np.isnan(value)]
    missing_values_position = [x['position'] for x in data_plot_missing_values]
    missing_values_previous_value = [x['value'] for x in data_plot_missing_values]

    plt.scatter(missing_values_position, missing_values_previous_value, s=dot_size, color='red', label='Valor Faltante')

    imputed_values = [dataset_imputed_values[i] for i in missing_values_position]

    plt.scatter(missing_values_position, imputed_values, s=dot_size, color='#268e32', label='Valor Imputado')

    # plt.title("Dia 200")

    plt.ylabel('Valor medido pelo sensor 5')
    plt.xlabel('Número da observação')

    plt.legend()
    plt.show()


def extract_features(dataset_path, missing_data_percentage=0, imputation_strategy=None):
    """
        %% Stand-alone Feature Extraction
        % Extrai as features do sinal original e aproximado. Sendo que o sinal
        % aproximado já está disponível em algum diretório. As features serão
        % salvas sobrescrevendo os arquivos já existentes para cada limiar de cada
        % algoritmo. Este script é útil caso se deseje alterar o cálculo das
        % features, bem como incluir outras.
    """

    # Qual sensor usaremos para comprimir (há 8 sensores no total).
    # COLUNA    1   2   3   4   5   6   7   8
    # SENSOR    3   5   6   7   10  12  14  16
    canal = 1

    temp = scipy.io.loadmat(dataset_path)

    # Obtém apenas a matriz desta observação com os 8 canais.
    dataset_orig = temp['data']

    # Obtém as leituras do canal de interesse.
    dataset_orig = dataset_orig[:, canal]

    # plt.scatter(range(0, len(dataset_orig)), dataset_orig, s=1)
    # plt.title("Dia 200")
    # plt.legend()
    # plt.show()

    # Normalização. (Let's hope this won't break the script).
    dataset_orig = normalize_var(dataset_orig, -1, 1)

    dataset_missing_data = simulate_missing_data(dataset_orig, missing_data_percentage)

    if imputation_strategy:
        new_dataset = imputation_strategy.impute_data(dataset_missing_data)
    else:
        new_dataset = realizar_amputacao_dados(dataset_missing_data)

    # Extrai as features do sinal original.
    feat_vector_orig = step_extraction(new_dataset, 1)

    plot_comparison_imputed_values_with_missing_values(dataset_orig, dataset_missing_data, new_dataset)

    return feat_vector_orig


def extract_features_for_knn(missing_data_percentage, files):
    """
        %% Stand-alone Feature Extraction
        % Extrai as features do sinal original e aproximado. Sendo que o sinal
        % aproximado já está disponível em algum diretório. As features serão
        % salvas sobrescrevendo os arquivos já existentes para cada limiar de cada
        % algoritmo. Este script é útil caso se deseje alterar o cálculo das
        % features, bem como incluir outras.
    """

    # Qual sensor usaremos para comprimir (há 8 sensores no total).
    # COLUNA    1   2   3   4   5   6   7   8
    # SENSOR    3   5   6   7   10  12  14  16
    canal = 1

    files_directory = '/home/alessandro/FELIPE/Z24_bridge/Z24Date/'

    full_database_with_missing_data = []
    full_database = []

    for j, filename in enumerate(files):
        print(filename)
        temp = scipy.io.loadmat(files_directory + filename)

        # Obtém apenas a matriz desta observação com os 8 canais.
        dataset_orig = temp['data']

        # Obtém as leituras do canal de interesse.
        dataset_orig = dataset_orig[:, canal]

        # Normalização. (Let's hope this won't break the script).
        dataset_orig = normalize_var(dataset_orig, -1, 1)

        full_database.append(dataset_orig)

        dataset_missing_data = simulate_missing_data(dataset_orig, missing_data_percentage)

        full_database_with_missing_data.append(dataset_missing_data)

    print('>> Imputation started... ')

    full_database_data_imputed = KNNImputation.impute_data(full_database_with_missing_data)

    print('>> Finished... ')

    dia_escolhido = 5
    dia_aleatorio = full_database[dia_escolhido]

    dot_size = 10

    plt.scatter(range(0, len(dia_aleatorio)), dia_aleatorio, s=dot_size, color='#26539b', label='Valor Inalterado')

    data_plot_missing_values = [{'position': i, 'value': dia_aleatorio[i]} for i, value in
                                enumerate(full_database_with_missing_data[dia_escolhido]) if np.isnan(value)]
    missing_values_position = [x['position'] for x in data_plot_missing_values]
    missing_values_previous_value = [x['value'] for x in data_plot_missing_values]

    print(data_plot_missing_values)

    plt.scatter(missing_values_position, missing_values_previous_value, s=dot_size, color='red', label='Valor Faltante')

    imputed_values = [full_database_data_imputed[dia_escolhido][i] for i in missing_values_position]

    print(imputed_values)

    plt.scatter(missing_values_position, imputed_values, s=dot_size, color='#268e32', label='Valor Imputado (Interpolação)')

    plt.ylabel('Valor medido pelo sensor 5')
    plt.xlabel('Número da observação')
    plt.legend()
    plt.show()

    extracted_features = []

    for row in full_database_data_imputed:
        # Extrai as features do sinal original.
        feat_vector_orig = step_extraction(row, 1)

        extracted_features.append(feat_vector_orig)

    return extracted_features
