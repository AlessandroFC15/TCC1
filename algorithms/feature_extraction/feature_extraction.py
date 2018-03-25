import random
import scipy.io

from algorithms.feature_extraction.step_extraction import step_extraction


def realizar_amputacao_dados(data, missing_data_percentage):
    num_samples_to_be_removed = round(len(data) * (missing_data_percentage / 100))

    new_database = list(data)

    for i in range(0, num_samples_to_be_removed):
        rand_num = random.randint(0, len(new_database) - 1)

        del new_database[rand_num]

    return new_database


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


def extract_features(dataset_path, missing_data_percentage=0):
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

    # Normalização. (Let's hope this won't break the script).
    dataset_orig = normalize_var(dataset_orig, -1, 1)

    dataset_orig = realizar_amputacao_dados(dataset_orig, missing_data_percentage)

    # Extrai as features do sinal original.
    feat_vector_orig = step_extraction(dataset_orig, 1)

    return feat_vector_orig
