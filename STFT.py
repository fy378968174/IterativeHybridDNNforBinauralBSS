# STFT and ISTFT are from the following link
# https://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python
import scipy

def stft(x, nfft=1024, hopsize=512):
    # w = scipy.hanning(nfft)
    w = scipy.hamming(nfft)
    X = scipy.array([scipy.fft(w*x[i:i+nfft])
                     for i in range(0, len(x)-nfft, hopsize)])
    return X

def istft(X, nfft=1024, hopsize=512):
    N = (X.shape[0]-1)*hopsize + nfft
    x = scipy.zeros(N)
    for n, i in enumerate(range(0, len(x)-nfft, hopsize)):
        x[i:i+nfft] += scipy.real(scipy.ifft(X[n]))
    return x
