import numpy as np
import matplotlib.pyplot as plt
from STFT import istft
import scipy.io.wavfile
import scipy.io as sio
import re

class RecoverData:
    def __init__(self, normParamsDir = '/ATTENTION_CHANGE_DIR_HERE/Params/RoomCmyNormParamsMixture.mat', overlap = 0.5, Fs = 8000):
        self.normParamsDir = normParamsDir
        self.overlap = overlap
        self.Fs = Fs

        if normParamsDir is not None:
            mat2 = sio.loadmat(normParamsDir)
            sig_std = mat2['sig_std']  # NFFT/2+1 x 1
            sig_mean = mat2['sig_mean']  # NFFT/2+1 x 1
            sig_std = sig_std[1:-1]
            sig_mean = sig_mean[1:-1]
            self.sig_std = sig_std.reshape(sig_std.size, )
            self.sig_mean = sig_mean.reshape(sig_mean.size, )



    def recover_from_spectrum(self, output, output_angle, savename = 'test.wav'): # size NFFT/2-1 * samples
        if self.normParamsDir is not None:
            # The original spectrum reverse the pre-whitening
            for k in range(output.shape[0]):
                output[k] *= self.sig_std
                output[k] += self.sig_mean

        # now apply istft
        # from log-spectrum to the magnitude
        # output = np.sqrt(np.power(10, output))  # get the magnitude
        output = np.sqrt(np.exp(output))  # get the magnitude
        output = np.multiply(output, np.exp(1j * output_angle))
        output2 = output[:, ::-1].conj()
        tempzero = np.zeros((output.shape[0],1))
        output = np.concatenate((tempzero, output, tempzero, output2), axis=1)
        # plt.figure(101)
        # plt.pcolor(np.abs(output[0:300,:].transpose()))
        # plt.show()

        NFFT = output.shape[1]
        overlap = int(self.overlap * NFFT)
        x = istft(output, NFFT, overlap)
        # plt.figure(102)
        # plt.title('Enhanced signal Wave...')
        # plt.plot(x)

        if savename is not None:
            scaled = np.int16(x / np.max(np.abs(x)) * 32767)
            scipy.io.wavfile.write(savename, self.Fs, scaled)

        return x

    def recover_from_spectrumOhio(self, output, output_angle, savename='test.wav'):  # size NFFT/2-1 * samples
        if self.normParamsDir is not None:
            # The original spectrum reverse the pre-whitening
            for k in range(output.shape[0]):
                output[k] *= self.sig_std
                output[k] += self.sig_mean



        # now apply istft
        # from log-spectrum to the magnitude
        # output = np.sqrt(np.power(10, output))  # get the magnitude
        output = np.sqrt(np.exp(output))  # get the magnitude
        output = np.multiply(output, np.exp(1j * output_angle))
        output2 = output[:, ::-1].conj()
        tempzero = np.zeros((output.shape[0], 1))
        output = np.concatenate((tempzero, output, tempzero, output2), axis=1)
        # plt.figure(101)
        # plt.pcolor(np.abs(output[0:200,:].transpose()))
        # plt.show()

        NFFT = output.shape[1]
        overlap = 160
        # x = istft(output, NFFT, overlap)
        N = (output.shape[0] - 1) * 160 + 512
        x = scipy.zeros(N)
        for n, i in enumerate(range(0, len(x) - 512, 160)):
            temp = scipy.real(scipy.ifft(output[n]))
            temp[320::] = 0.0
            x[i:i + 512] += temp

        # plt.figure(102)
        # plt.title('Enhanced signal Wave...')
        # plt.plot(x)

        if savename is not None:
            scaled = np.int16(x / np.max(np.abs(x)) * 32767)
            scipy.io.wavfile.write(savename, self.Fs, scaled)

        return x

    def istftOhio(self,X):
        N = (X.shape[0] - 1) * 160 + 512
        x = scipy.zeros(N)
        for n, i in enumerate(range(0, len(x) - 512, 160)):
            temp = scipy.real(scipy.ifft(X[n]))
            temp[320::] = 0.0
            x[i:i + 512] += temp
        return x






def showData(y_train,mixture,output): # (NFFT/2-1)*2 * samples, (NFFT/2-1) * samples, 2 * (NFFT/2-1) * samples
    plt.figure(99).suptitle('source1, source2, s1estimate, s2estimate')
    ax1 = plt.subplot(231)
    im1 = ax1.pcolor(y_train[0:255, :])
    plt.colorbar(im1)
    ax2 = plt.subplot(232, sharex=ax1)
    im2 = ax2.pcolor(y_train[255:510, :])
    plt.colorbar(im2)
    ax3 = plt.subplot(234, sharex=ax1)
    im3 = ax3.pcolor(output[0, :, :].squeeze())
    plt.colorbar(im3)
    ax4 = plt.subplot(235, sharex=ax1)
    im4 = ax4.pcolor(output[1, :, :].squeeze())
    plt.colorbar(im4)

    ax5 = plt.subplot(233, sharex=ax1)
    im5 = ax5.pcolor(mixture)
    plt.colorbar(im5)


    plt.show()  # to force matlab plot shows up http://stackoverflow.com/questions/9280171/matplotlib-python-show-returns-immediately




def formatData(X_train, y_train, dg):
    new_shape = (X_train.shape[0], 1, dg.halfNFFTtrim, dg.batchLen)
    X_train_spectrum = X_train[:, 0:dg.halfNFFTtrim, :]
    X_train_spectrum = X_train_spectrum.reshape(new_shape)
    X_train_IPD1 = X_train[:, dg.halfNFFTtrim:dg.halfNFFTtrim * 2, :]
    X_train_IPD1 = X_train_IPD1.reshape(new_shape)
    X_train_IPD2 = X_train[:, dg.halfNFFTtrim * 2:dg.halfNFFTtrim * 3, :]
    X_train_IPD2 = X_train_IPD2.reshape(new_shape)
    X_train_new = np.concatenate((X_train_spectrum, X_train_IPD1, X_train_IPD2), axis=1)

    new_shape2 = (X_train.shape[0], 1, dg.halfNFFTtrim, 1)
    y1_train = y_train[:, 0:dg.halfNFFTtrim]
    y1_train = y1_train.reshape(new_shape2)
    y2_train = y_train[:, dg.halfNFFTtrim:dg.halfNFFTtrim * 2]
    y2_train = y2_train.reshape(new_shape2)
    y_train_new = np.concatenate((y1_train, y2_train), axis=1)

    return (X_train_new,y_train_new)


def formatDataRaw(X_train, y_train, dg):
    new_shape = (X_train.shape[0], 1, dg.halfNFFTtrim, dg.batchLen)
    X_train_spectrum = X_train[:, 0:dg.halfNFFTtrim, :]
    X_train_spectrum = X_train_spectrum.reshape(new_shape)
    X_train_IPD = X_train[:, dg.halfNFFTtrim:dg.halfNFFTtrim * 2, :]
    X_train_IPD = X_train_IPD.reshape(new_shape)
    X_train_new = np.concatenate((X_train_spectrum, X_train_IPD), axis=1)

    new_shape2 = (X_train.shape[0], 1, dg.halfNFFTtrim, 1)
    y1_train = y_train[:, 0:dg.halfNFFTtrim]
    y1_train = y1_train.reshape(new_shape2)
    y2_train = y_train[:, dg.halfNFFTtrim:dg.halfNFFTtrim * 2]
    y2_train = y2_train.reshape(new_shape2)
    y_train_new = np.concatenate((y1_train, y2_train), axis=1)

    return (X_train_new,y_train_new)



def tauInit(mat, filename):
    batchLen = 11
    halfBatchLen = 5

    # s1LogPower = mat['s1LogPower']
    # s1LogPower = np.asarray(s1LogPower)  # [:, Index]
    # # s1LogPower = s1LogPower[:, 0:self.batchLen]
    #
    # s2LogPower = mat['s2LogPower']
    # s2LogPower = np.asarray(s2LogPower)
    # # s2LogPower = s2LogPower[:, 0:self.batchLen]

    mixAngle_L = mat['mixAngle_L']
    mixAngle_L = np.asarray(mixAngle_L)
    # mixAngle_L = mixAngle_L[:, 0:self.batchLen]

    mixAngle_R = mat['mixAngle_R']
    mixAngle_R = np.asarray(mixAngle_R)
    # mixAngle_R = mixAngle_R[:, 0:self.batchLen]

    mixLogPower = mat['mixLogPower']
    mixLogPower = np.asarray(mixLogPower)
    # mixLogPower = mixLogPower[:, 0:self.batchLen]

    mixAngle_LR = mixAngle_L[1:-1, :] - mixAngle_R[1:-1, :]
    # # map the angle different between [-pi pi]
    # mixAngle_LR[mixAngle_LR > np.pi] = mixAngle_LR[mixAngle_LR > np.pi] - 2 * np.pi
    # mixAngle_LR[mixAngle_LR < -np.pi] = mixAngle_LR[mixAngle_LR < -np.pi] + 2 * np.pi
    normGCC = np.exp(1j*mixAngle_LR)


    # np.exp(1j*)
    N = 41
    Fs = 16000
    tau_candi = np.linspace(-0.001, 0.001, num=N)
    f_array1 = np.linspace(0,Fs/2, 257)
    f_array = f_array1[1:-1]
    f_array = f_array.reshape((f_array.size, 1))
    c_array = np.zeros((N,))

    # try to use only half of the values
    medianV = np.median(mixLogPower)
    useFlag = np.greater(mixLogPower,medianV)

    for i in range(N):
        tau = tau_candi[i]
        ejwt = np.exp(1j*2*np.pi*f_array*tau)
        # b.reshape((b.size, 1))
        tmp = np.multiply(normGCC,ejwt)
        tmp = np.multiply(tmp,useFlag[1:-1,:])
        c_array[i] = np.mean(tmp)

    # plt.figure(99).suptitle('GCC-PHAT over candidate delays')
    # plt.plot(tau_candi,c_array)
    # plt.grid(True)
    # plt.ylabel('tau')
    # plt.ylabel('GCC')
    # # plt.show()

    # Find the two peaks as the two initial taus associated with the targets
    # first peak
    ind1 = np.argmax(c_array)
    c_array[max(ind1-3,0):min(ind1+3,N)]=0

    # plt.plot(tau_candi, c_array)
    # plt.show()

    # second peak
    ind2 = np.argmax(c_array)

    # signal more to the left side comes out first
    if ind1>ind2:
        tt = ind1
        ind1 = ind2
        ind2 = tt

    peak_tau_array = np.array([tau_candi[ind1],tau_candi[ind2]])
    tmp1 = -2 * np.pi * f_array * tau_candi[ind1]
    tmp1 = tmp1.reshape((1, tmp1.size))
    tmp1 = (tmp1 + np.pi) % (2 * np.pi) - np.pi

    tmp2 = -2 * np.pi * f_array * tau_candi[ind2]
    tmp2 = tmp2.reshape((1, tmp2.size))
    tmp2 = (tmp2 + np.pi) % (2 * np.pi) - np.pi

    IPD_init_array = np.concatenate((tmp1,tmp2),0)

    return (peak_tau_array, IPD_init_array, mixAngle_LR, mixLogPower)




def refineIPDmean(mixAngle_LR, mixLogPower, output, peak_tau_array, IPD_mean_est):

    normGCC = np.exp(1j * mixAngle_LR)
    normGCC = normGCC[:, 5:-5]

    # try to use only half of the values
    medianV = np.median(mixLogPower)
    useFlag1 = np.greater(mixLogPower[1:-1, 5:-5], medianV)

    useFlag2 = np.greater(output[:, 0, :, 0], output[:, 1, :, 0]).transpose()

    N = 11
    Fs = 16000
    f_array1 = np.linspace(0, Fs / 2, 257)
    f_array = f_array1[1:-1]
    f_array = f_array.reshape((f_array.size, 1))
    peak_tau_array_new = np.array([0.0, 0.0])
    for index in range(2):
        tau_candi = peak_tau_array[index]+ np.linspace(-0.00005, 0.00005, num=N)

        c_array = np.zeros((N,))

        medianV = np.median(output[:, index, :, 0])
        useFlag3 = (np.greater(output[:, index, :, 0], medianV)).transpose()

        if index==0:
            useFlag = np.logical_and(np.logical_and(useFlag1,useFlag2),useFlag3)
        else:
            useFlag = np.logical_and(np.logical_and(useFlag1, np.logical_not(useFlag2)), useFlag3)

        for i in range(N):
            tau = tau_candi[i]
            ejwt = np.exp(1j * 2 * np.pi * f_array * tau)
            # b.reshape((b.size, 1))
            tmp = np.multiply(normGCC, ejwt)
            tmp = np.multiply(tmp, useFlag)
            c_array[i] = np.sum(tmp) / np.sum(useFlag)

        ind = np.argmax(c_array)
        peak_tau_array_new[index] = tau_candi[ind]

    #refine the IPD mean
    tmp1 = -2 * np.pi * f_array * peak_tau_array_new[0]
    tmp1 = tmp1.reshape((1, tmp1.size))
    tmp1 = (tmp1 + np.pi) % (2 * np.pi) - np.pi

    tmp2 = -2 * np.pi * f_array * peak_tau_array_new[1]
    tmp2 = tmp2.reshape((1, tmp2.size))
    tmp2 = (tmp2 + np.pi) % (2 * np.pi) - np.pi

    IPD_mean_est_new = np.concatenate((tmp1, tmp2), 0)
    # IPD_mean_est_new = IPD_mean_est
    IPD_mean_est_new2 = IPD_mean_est_new.copy()

    for index in range(2):
        tmpIPDmean = IPD_mean_est_new[index].reshape((255,1))
        tmpmixAngle_LR = mixAngle_LR-tmpIPDmean
        tmpmixAngle_LR[tmpmixAngle_LR > np.pi] = tmpmixAngle_LR[tmpmixAngle_LR > np.pi] - 2 * np.pi
        tmpmixAngle_LR[tmpmixAngle_LR < -np.pi] = tmpmixAngle_LR[tmpmixAngle_LR < -np.pi] + 2 * np.pi
        if index == 0:
            useFlag = useFlag2#np.logical_and(useFlag1, useFlag2)
        else:
            useFlag = np.logical_not(useFlag2)#np.logical_and(useFlag1, np.logical_not(useFlag2))
        threshold = 1.0
        useFlag3 = (np.less(np.abs(tmpmixAngle_LR[:, 5:-5]), threshold))
        useFlag = np.logical_and(useFlag,useFlag3)

        for f in range(255):
            useFlagF = useFlag[f]
            tmpmixAngle_LRF = tmpmixAngle_LR[f,5:-5]
            tmp = np.multiply(tmpmixAngle_LRF, useFlagF)
            useNumber = np.sum(useFlagF)
            if useNumber>10: #int(useFlagF.size/8):
                angleShift = np.sum(tmp) / useNumber
            else:
                angleShift = 0
            IPD_mean_est_new2[index,f] += angleShift

    # a = 0
    # plt.figure(123)
    # plt.plot(IPD_mean_est_new2.transpose())
    # plt.plot(IPD_mean_est_new.transpose())
    # plt.show()


    return (peak_tau_array_new, IPD_mean_est_new2)



def extractInputFeature(mat, filename, idealFlag=False, IPD_mean_est=None):
    # The mean and variance of the features to normalise the input and output feature vector
    mat2 = sio.loadmat('/ATTENTION_CHANGE_DIR_HERE/Params/RoomCmyNormParamsMixture.mat')
    sig_std = mat2['sig_std']  # Dimx1
    sig_mean = mat2['sig_mean']  # Dimx1
    sig_std = sig_std.reshape(sig_std.size, )
    sig_mean = sig_mean.reshape(sig_std.size, )

    batchLen = 11
    halfBatchLen = 5

    s1LogPower = mat['s1LogPower']
    s1LogPower = np.asarray(s1LogPower)  # [:, Index]
    # s1LogPower = s1LogPower[:, 0:self.batchLen]

    s2LogPower = mat['s2LogPower']
    s2LogPower = np.asarray(s2LogPower)
    # s2LogPower = s2LogPower[:, 0:self.batchLen]

    mixAngle_L = mat['mixAngle_L']
    mixAngle_L = np.asarray(mixAngle_L)
    # mixAngle_L = mixAngle_L[:, 0:self.batchLen]

    mixAngle_R = mat['mixAngle_R']
    mixAngle_R = np.asarray(mixAngle_R)
    # mixAngle_R = mixAngle_R[:, 0:self.batchLen]

    mixLogPower = mat['mixLogPower']
    mixLogPower = np.asarray(mixLogPower)
    # mixLogPower = mixLogPower[:, 0:self.batchLen]

    mixAngle_LR = mixAngle_L[1:-1, :] - mixAngle_R[1:-1, :]
    # map the angle different between [-pi pi]
    mixAngle_LR[mixAngle_LR > np.pi] = mixAngle_LR[mixAngle_LR > np.pi] - 2 * np.pi
    mixAngle_LR[mixAngle_LR < -np.pi] = mixAngle_LR[mixAngle_LR < -np.pi] + 2 * np.pi

    shiftAng1 = mixAngle_LR.copy()
    shiftAng2 = mixAngle_LR.copy()

    if idealFlag:
        Azimuth_array = [-60,-30,0,30,60]
        Nums = list(map(int, re.findall('[+-]?\d+', filename)))
        azi1_index = Azimuth_array.index(Nums[0])  # np.where(self.Azimuth_array == Nums[0])[0][0]
        azi2_index = Azimuth_array.index(Nums[1])  # angel of the second speaker
        # shiftAng1 = [mixAngle_LR[:,k]-self.IPD_mean[azi1_index] for k in range(mixLogPower.shape[1])]

        mat1 = sio.loadmat('/ATTENTION_CHANGE_DIR_HERE/Params/RoomCGMMParams.mat')
        IPD_mean = mat1['IPD_mean']
        IPD_mean = np.asarray(IPD_mean).transpose()
        for k in range(mixLogPower.shape[1]):
            shiftAng1[:, k] -= IPD_mean[azi1_index]
            shiftAng2[:, k] -= IPD_mean[azi2_index]
    else:
        for k in range(mixLogPower.shape[1]):
            shiftAng1[:, k] -= IPD_mean_est[0]
            shiftAng2[:, k] -= IPD_mean_est[1]

    shiftAng1[shiftAng1 > np.pi] = shiftAng1[shiftAng1 > np.pi] - 2 * np.pi
    shiftAng1[shiftAng1 < -np.pi] = shiftAng1[shiftAng1 < -np.pi] + 2 * np.pi
    shiftAng2[shiftAng2 > np.pi] = shiftAng2[shiftAng2 > np.pi] - 2 * np.pi
    shiftAng2[shiftAng2 < -np.pi] = shiftAng2[shiftAng2 < -np.pi] + 2 * np.pi

    shiftAng1 = np.power(shiftAng1, 2)
    shiftAng2 = np.power(shiftAng2, 2)

    shiftAng1 = np.exp(-shiftAng1)
    shiftAng2 = np.exp(-shiftAng2)

    # normalise the data(prewhitening)
    for k in range(mixLogPower.shape[1]):
        s1LogPower[:, k] -= sig_mean
        s1LogPower[:, k] /= sig_std
        s2LogPower[:, k] -= sig_mean
        s2LogPower[:, k] /= sig_std
        mixLogPower[:, k] -= sig_mean
        mixLogPower[:, k] /= sig_std


    N = mixLogPower.shape[1] - batchLen+1
    chooseIndex = np.arange(N)  # (range(0, N)

    # concatenate the feature as the input
    Index1 = (np.tile(range(0, batchLen), (N, 1))).transpose()
    Index2 = np.tile(chooseIndex, (batchLen, 1))
    Index = Index1 + Index2
    mixLogPower = mixLogPower[:, Index]

    # tmp = np.reshape(tmp, (self.Dim_in, N), order="F")
    mixLogPower = np.transpose(mixLogPower, (2, 0, 1))

    shiftAng1 = shiftAng1[:, Index]
    shiftAng1 = np.transpose(shiftAng1, (2, 0, 1))

    shiftAng2 = shiftAng2[:, Index]
    shiftAng2 = np.transpose(shiftAng2, (2, 0, 1))

    currentBatchDataIn = np.concatenate((mixLogPower[:, 1:-1, :], shiftAng1, shiftAng2), axis=1)
    currentBatchDataOut = (np.concatenate((s1LogPower[1:-1, chooseIndex+halfBatchLen], s2LogPower[1:-1, chooseIndex+halfBatchLen]), axis=0)).transpose()


    return (currentBatchDataIn, currentBatchDataOut, mixAngle_L[1:-1, 5:-5])


def extractInputFeatureRaw(mat, filename):
    # The mean and variance of the features to normalise the input and output feature vector
    mat2 = sio.loadmat('/ATTENTION_CHANGE_DIR_HERE/Params/RoomCmyNormParamsMixture.mat')
    sig_std = mat2['sig_std']  # Dimx1
    sig_mean = mat2['sig_mean']  # Dimx1
    sig_std = sig_std.reshape(sig_std.size, )
    sig_mean = sig_mean.reshape(sig_std.size, )

    batchLen = 11
    halfBatchLen = 5

    s1LogPower = mat['s1LogPower']
    s1LogPower = np.asarray(s1LogPower)  # [:, Index]
    # s1LogPower = s1LogPower[:, 0:self.batchLen]

    s2LogPower = mat['s2LogPower']
    s2LogPower = np.asarray(s2LogPower)
    # s2LogPower = s2LogPower[:, 0:self.batchLen]

    mixAngle_L = mat['mixAngle_L']
    mixAngle_L = np.asarray(mixAngle_L)
    # mixAngle_L = mixAngle_L[:, 0:self.batchLen]

    mixAngle_R = mat['mixAngle_R']
    mixAngle_R = np.asarray(mixAngle_R)
    # mixAngle_R = mixAngle_R[:, 0:self.batchLen]

    mixLogPower = mat['mixLogPower']
    mixLogPower = np.asarray(mixLogPower)
    # mixLogPower = mixLogPower[:, 0:self.batchLen]

    mixAngle_LR = mixAngle_L[1:-1, :] - mixAngle_R[1:-1, :]
    # map the angle different between [-pi pi]
    mixAngle_LR[mixAngle_LR > np.pi] = mixAngle_LR[mixAngle_LR > np.pi] - 2 * np.pi
    mixAngle_LR[mixAngle_LR < -np.pi] = mixAngle_LR[mixAngle_LR < -np.pi] + 2 * np.pi


    # normalise the data(prewhitening)
    for k in range(mixLogPower.shape[1]):
        s1LogPower[:, k] -= sig_mean
        s1LogPower[:, k] /= sig_std
        s2LogPower[:, k] -= sig_mean
        s2LogPower[:, k] /= sig_std
        mixLogPower[:, k] -= sig_mean
        mixLogPower[:, k] /= sig_std


    N = mixLogPower.shape[1] - batchLen+1
    chooseIndex = np.arange(N)  # (range(0, N)

    # concatenate the feature as the input
    Index1 = (np.tile(range(0, batchLen), (N, 1))).transpose()
    Index2 = np.tile(chooseIndex, (batchLen, 1))
    Index = Index1 + Index2
    mixLogPower = mixLogPower[:, Index]

    # tmp = np.reshape(tmp, (self.Dim_in, N), order="F")
    mixLogPower = np.transpose(mixLogPower, (2, 0, 1))

    mixAngle_LR = mixAngle_LR[:, Index]
    mixAngle_LR = np.transpose(mixAngle_LR, (2, 0, 1))

    currentBatchDataIn = np.concatenate((mixLogPower[:, 1:-1, :], mixAngle_LR), axis=1)
    currentBatchDataOut = (np.concatenate((s1LogPower[1:-1, chooseIndex+halfBatchLen], s2LogPower[1:-1, chooseIndex+halfBatchLen]), axis=0)).transpose()


    return (currentBatchDataIn, currentBatchDataOut, mixAngle_L[1:-1, 5:-5])
