import numpy as np
import numpy.matlib
from scipy import signal


def nextpow2(i):
    i = abs(i)

    n = 0

    while True:
        if 2 ** n >= i:
            return n

        n += 1


def fct_fft(x, dt):
    try:
        nrow, ncol = x.shape  # matrix dimensions
    except ValueError:
        nrow, = x.shape
        ncol = 1

    # t = (0:nrow - 1)*dt # Time vector
    t = [x * dt for x in range(0, nrow)]  # Time vector
    Fs = 1 / dt  # Sampling frequency

    # Exponential window for accelerometers
    hann_window = np.hanning(nrow)

    m = x * hann_window

    NFFT = 2 ** nextpow2(nrow)  # Next power of 2 from length of y(t)

    freq = Fs / 2 * np.linspace(0, 1, NFFT / 2)  # Nyquist frequency

    AUX = np.fft.fft(m, NFFT) / nrow

    yfft = AUX[0:int(NFFT / 2)]

    return freq, yfft


def feature_shm_ufpa(data, dt, limits):
    # dt data increment in seconds
    fs = 1 / dt  # frequency

    # Application of a filter

    # % Design a Butterworth IIR digital filter
    # For data sampled at 100 Hz, design a 3th-order highpass Butterworth
    # filter, with a cutoff frequency of 3 Hz, which corresponds to a
    # normalized value of 0.06.

    wn = 2 / (fs / 2)  # normalized cutoff frequency Wn (2 Hz)

    # [bb, aa] = butter(3, wn, 'high')
    bb, aa = signal.butter(3, wn, 'highpass')

    # Performs zero-phase digital filtering by processing the input data
    data = signal.filtfilt(bb, aa, data)

    # clear aa bb
    del bb
    del aa

    #################################
    # Estimate the FFT

    # Hanning window
    xfreq, Yfft = fct_fft(data, dt)

    m = len(xfreq)

    df = xfreq[1] - xfreq[0]

    # Auto-power spectral density
    psd = 1 / (m * dt) * Yfft * Yfft.conj()

    #################################
    # Averaged normalized power spectral density - ANPSD

    # Estimate ANPSD

    mean_vector = np.sum(psd)

    # Reshape necessário para que tenha o mesmo shape que o resultado do repmat no mean_vector
    psd = numpy.reshape(psd, (32768, 1))

    NPSD = psd / numpy.matlib.repmat(mean_vector, m, 1)

    ANPSD = np.sum(NPSD, axis=1)

    #################################
    # Extract the natural frequencies
    a1 = limits[0][0]
    d1 = limits[0][1]

    indx = np.argmax(abs(ANPSD[a1 - d1 - 1: a1 + d1]))
    ind = a1 - d1 + indx - 1
    freq1 = xfreq[ind]

    a1 = limits[1][0]
    d1 = limits[1][1]

    indx = np.argmax(abs(ANPSD[a1 - d1 - 1: a1 + d1]))
    ind = a1 - d1 + indx - 1
    freq2 = xfreq[ind]

    energy = np.trapz(abs(ANPSD[round(len(ANPSD) / 2):len(ANPSD)]))

    return [freq1, freq2, energy]


def step_extraction(data, col):
    """
    STEP_EXTRACTION Etapa de extração de features.

%   São utilizadas 3 features do sinal para a detecção de danos:
%   As duas primeiras são a Primeira e Terceira Frequência Natural (F1 e
%   F3, respectivamente).
%   A terceira consiste na quantidade de energia presente na segunda metade
%   do espectro.

%   INPUT
%       data: Matrix com as leitura dos acelerômetros no domínio do tempo.
%       col: Coluna da matriz com as leituras desejadas. Nos dados
%       originais, col=2 equivale às leituras do sensor 5. Nos dados
%       comprimidos, normalmente col=1, pois se comprimiu as leituras de
%       apenas um sensor.

%   OUTPUT
        feat_vector: Feature vector com as 3 features (F1, F3, A).
    """

    # Propriedades de amostragem
    dt = 0.01  # segundos

    # Janelas para estimar as frequências naturais F1 e F3.
    limits = [[2600, 300], [6600, 900]]

    data_coluna = data  # Pegar data só de uma coluna

    feat_vector = feature_shm_ufpa(data_coluna, dt, limits)

    return feat_vector
