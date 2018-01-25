import scipy.io
import numpy


def countNumberNullInstances(array):
    return dict(zip(*numpy.unique(array, return_counts=True)))


mat = scipy.io.loadmat('database/Z24Full2.mat')

print(mat['Z24_TDS'].size)
print(mat['Z24_TDS'])

mat['Z24_TDS'][0] = 'None'
mat['Z24_TDS'][1] = 0

print(countNumberNullInstances(mat['Z24_TDS']))
print(countNumberNullInstances(mat['Z24_TDS'])[0])
print(countNumberNullInstances(mat['Z24_TDS'])['None'])