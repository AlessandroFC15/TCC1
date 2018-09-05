import ctypes

import numpy as np
from numpy.ctypeslib import ndpointer

lib = ctypes.cdll.LoadLibrary('./compression_dll.dll')

# ==========
# APCA
# ==========
APCA_Run = lib.APCA_Run
APCA_Run.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                     ctypes.c_size_t,
                     ctypes.c_float,
                     ctypes.c_int,
                     ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
APCA_Run.restype = ctypes.c_double


def APCA_Compression(data, eps, relative_eps=False):
    out_data = np.empty(data.shape)
    error_type = 1 if relative_eps else 0
    ratio = APCA_Run(data, data.size, eps, error_type, out_data)
    return out_data, ratio


# ==========
# PCA
# ==========

PCA_Run = lib.PCA_Run

PCA_Run.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                    ctypes.c_size_t,
                    ctypes.c_float,
                    ctypes.c_int,
                    ctypes.c_int,
                    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]

PCA_Run.restype = ctypes.c_double


def PCA_Compression(data, eps, relative_eps=False):
    out_data = np.empty(data.shape)
    error_type = 1 if relative_eps else 0
    ratio = PCA_Run(data, data.size, eps, error_type, 16, out_data)
    return out_data, ratio


# ==========
# Slide Filter
# ==========
SF_Run = lib.SF_Run

SF_Run.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                   ctypes.c_size_t,
                   ctypes.c_float,
                   ctypes.c_int,
                   ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]

SF_Run.restype = ctypes.c_double


def SF_Compression(data, eps, relative_eps=False):
    out_data = np.empty(data.shape)
    error_type = 1 if relative_eps else 0
    ratio = SF_Run(data, data.size, eps, error_type, out_data)
    return out_data, ratio


# ==========
# PWHLH
# ==========
PWLH_Run = lib.PWLH_Run

PWLH_Run.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                     ctypes.c_size_t,
                     ctypes.c_float,
                     ctypes.c_int,
                     ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]

PWLH_Run.restype = ctypes.c_double


def PWLH_Compression(data, eps, relative_eps=False):
    out_data = np.empty(data.shape)
    error_type = 1 if relative_eps else 0
    ratio = PWLH_Run(data, data.size, eps, error_type, out_data)
    return out_data, ratio


# ==========
# CHEB NOT WORKING
# ==========
CHEB_Run = lib.CHEB_Run

CHEB_Run.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                     ctypes.c_size_t,
                     ctypes.c_float,
                     ctypes.c_int,
                     ctypes.c_int,
                     ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]

CHEB_Run.restype = ctypes.c_double

# ==========
# RMSE
# ==========
RMSE = lib.RMSE

RMSE.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                 ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                 ctypes.c_int]

RMSE.restype = ctypes.c_double


def RMSE_Error(data1, data2):
    assert data1.shape == data2.shape
    return RMSE(data1, data2, data1.size)


# ==========
# STDV
# ==========
STDV = lib.STDV

STDV.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                 ctypes.c_int]

STDV.restype = ctypes.c_double


def StandardDeviation(data1):
    return STDV(data1, data1.size)


if __name__ == "__main__":
    # indata = np.random.rand(input_size) * 2
    import scipy.io as spio  # version 0.17.0


    def valid(ind, outd, esp):
        for i in range(len(ind)):
            dif = abs(ind[i] - outd[i])
            if not dif <= esp:
                print("D", dif, (dif < esp))


    matfile = 'Z24Full2.mat'
    matdata = spio.loadmat(matfile)
    a = matdata['Z24_TWN'].T
    data = a[0]

    stdv = STDV(data, data.size)
    K = 1

    pca_taxas = np.zeros(50)
    apca_taxas = np.zeros(50)
    pwlh_taxas = np.zeros(50)
    sf_taxas = np.zeros(50)
    ks = np.zeros(50)

    mx = np.max(data)
    mn = np.min(data)

    print(stdv)

    for k in range(0, 50):
        K = k / 10
        # eps = stdv *  K / ( mx - mn)
        eps = K
        print(eps)

        pca_result = PCA_Compression(data, eps)
        apca_result = APCA_Compression(data, eps)
        pwlh_result = PWLH_Compression(data, eps)
        sf_result = SF_Compression(data, eps)

        pca_taxas[k] = 1 - pca_result[1]
        apca_taxas[k] = 1 - apca_result[1]
        pwlh_taxas[k] = 1 - pwlh_result[1]
        sf_taxas[k] = 1 - sf_result[1]
        ks[k] = K

    import matplotlib.pyplot as plt

    plt.plot(ks, pca_taxas, 'k')
    plt.plot(ks, apca_taxas, 'b')
    plt.plot(ks, pwlh_taxas, 'r')
    plt.plot(ks, sf_taxas, 'y')

    plt.axis([0, 5, 0, 5])
    plt.show()
