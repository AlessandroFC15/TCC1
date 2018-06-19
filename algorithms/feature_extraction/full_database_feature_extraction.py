import csv
import decimal
import time

import os
import re

from algorithms.feature_extraction.feature_extraction import extract_features


def save_features_to_csv_file(missing_data_percentage, num_iterations=1, imputation_strategy=None):
    files_directory = '/home/alessandro/FELIPE/Z24_bridge/Z24Date/'

    # Pegar todos os arquivos registrados às 12h
    files = [f for f in os.listdir(files_directory) if re.match(r'.*_12\.mat', f)]
    files.sort()

    for i in range(num_iterations):
        t0 = time.time()

        extracted_data = []

        for j, filename in enumerate(files):
            print('#{} | {} | Extracting {} ...'.format(i + 1, j + 1, filename))
            extracted_data.append(
                extract_features(files_directory + filename, missing_data_percentage, imputation_strategy))

        print(">> Criando arquivo csv...")

        file_name = '/home/alessandro/Documentos/Programming/Projects/TCC1/algorithms/features/{}' \
                    'Features_Originais_Hora_12_Sensor_5_MDP_{}_{}.csv'.format(
            ('{}/'.format(imputation_strategy.description) if imputation_strategy else ''), missing_data_percentage, i)

        with open(file_name, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')

            for row in extracted_data:
                csv_writer.writerow([round(x, 4) for x in row])

        t1 = time.time()
        total = t1 - t0

        print('Tempo passado: {}'.format(total))


def get_number_decimal_places_from_float(float_number):
    return decimal.Decimal(str(float_number)).as_tuple().exponent * -1
