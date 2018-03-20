from numpy import genfromtxt

from algoritmos_felipe.DamageDetection import K_Means


def get_extracted_data():
    current_file = '/home/alessandro/Documentos/Programming/Projects/TCC1/algorithms/feature_extraction/Features_Originais_Hora_12_Sensor_5.csv'
    data = genfromtxt(current_file, delimiter=',')

    # Pegando apenas as frequências
    data = data[:, [0, 1]]
    return data


all_data = get_extracted_data()

learn_data = all_data[1:158, :]
#
alg = K_Means(4)
alg.train_and_test(learn_data, all_data, 197)

print(alg.UCL)
print(alg.err)
print(alg.class_states)

# print(alg.err)
