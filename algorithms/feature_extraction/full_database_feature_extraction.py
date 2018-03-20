import csv
import decimal

import os
import re

from algorithms.feature_extraction.feature_extraction import extract_features


def get_number_decimal_places_from_float(float_number):
    return decimal.Decimal(str(float_number)).as_tuple().exponent * -1


files_directory = '/home/alessandro/FELIPE/Z24_bridge/Z24Date/'

# Pegar todos os arquivos registrados às 12h
files = [f for f in os.listdir(files_directory) if re.match(r'.*_12\.mat', f)]
files.sort()

extracted_data = []

for i, filename in enumerate(files):
    print('{} | Extracting {} ...'.format(i + 1, filename))
    extracted_data.append(extract_features(files_directory + filename))

print(">> Criando arquivo csv...")

with open('Features_Originais_Hora_12_Sensor_5.csv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')

    for row in extracted_data:
        csv_writer.writerow([round(x, 4) for x in row])


# f1 = [x[0] for x in extracted_data]
#
# axes = plt.gca()
# axes.set_ylim([3.5, 4.4])
#
# plt.scatter(range(0, len(f1)), f1, s=10)
# plt.show()
