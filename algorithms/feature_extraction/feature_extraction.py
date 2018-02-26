import os
import scipy.io

from algorithms.feature_extraction.step_extraction import step_extraction


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


"""
    %% Stand-alone Feature Extraction
    % Extrai as features do sinal original e aproximado. Sendo que o sinal
    % aproximado já está disponível em algum diretório. As features serão
    % salvas sobrescrevendo os arquivos já existentes para cada limiar de cada
    % algoritmo. Este script é útil caso se deseje alterar o cálculo das
    % features, bem como incluir outras.
"""

# Separador de arquivos da plataforma.
# f = filesep

# Usaremos todas as leituras feitas em uma determinada hora.
home_orig = '/home/alessandro/FELIPE/Z24_bridge/Z24Date'
hora = '12'  # Hora do monitoramento. Valores possíveis 00-23.
pattern = "*_{hora}.mat".format(hora=hora)  # Padrão para buscar leituras de uma determinada hora.

files_orig = dir(os.path.join(home_orig, pattern))

# Qual sensor usaremos para comprimir (há 8 sensores no total).
# COLUNA    1   2   3   4   5   6   7   8
# SENSOR    3   5   6   7   10  12  14  16
canal = 2

# Carrega os dados
temp = scipy.io.loadmat('/home/alessandro/FELIPE/Z24_bridge/Z24Date/data19971111_12.mat')

# Obtém apenas a matriz desta observação com os 8 canais.
dataset_orig = temp['data']

# Obtém as leituras do canal de interesse.
dataset_orig = dataset_orig[:, canal]

# Normalização. (Let's hope this won't break the script).
dataset_orig = normalize_var(dataset_orig, -1, 1)

# Extrai as features do sinal original.
feat_vector_orig = step_extraction(dataset_orig, 1)

print(feat_vector_orig)
