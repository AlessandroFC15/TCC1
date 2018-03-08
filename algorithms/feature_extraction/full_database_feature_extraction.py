import matplotlib.pyplot as plt
import os
import re

from algorithms.feature_extraction.feature_extraction import extract_features

files_directory = '/home/alessandro/FELIPE/Z24_bridge/Z24Date/'

# Pegar todos os arquivos registrados às 12h
files = [f for f in os.listdir(files_directory) if re.match(r'.*_12\.mat', f)]
files.sort()

extracted_data = []

for filename in files:
    print('Extracting {} ...'.format(filename))
    extracted_data.append(extract_features(files_directory + filename))

f1 = [x[0] for x in extracted_data]

axes = plt.gca()
axes.set_ylim([3.5, 4.4])

plt.scatter(range(0, len(f1)), f1, s=10)
plt.show()
