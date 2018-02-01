from keras import layers
from keras import models
from keras.utils import plot_model
from keras.optimizers import SGD
import scipy.io as sio

import numpy as np
from STFT import istft
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
import copy


from DataGeneration import dataGenBig
from Others import *
# from Others import formatData, extractInputFeature, extractInputFeatureRaw
from GenerateModels import generateModel, generateModelRaw, generateModelMLP, generateModelRawMLP


DataDir = "/ATTENTION_CHANGE_DIR_HERE/DLTestData/Unmatched/"
room = 'RoomC'
Azimuth_array = [-45,5,45]
MatchMode = 'Unmatched'


SpeakerN = 4

# The proposed model
model = generateModel()
# print(model.summary())
# plot_model(model, to_file='model.png')
# =========continue to train the model
optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=optim)
model.load_weights('NewResults/BSSmodelProcessFeature28Sept')
RawMode = False
MLPMode = False
app = 'Process'




class MyStruct():
    def __init__(self):
        self.halfNFFTtrim = 255
        self.batchLen = 11
dg = MyStruct()

params_dir = '/ATTENTION_CHANGE_DIR_HERE/Params/RoomCmyNormParamsMixture.mat'
save_dir = '/ATTENTION_CHANGE_DIR_HERE/DLResults/'

for genderFlag in ['MM', 'MF', 'FF']:
    dirName = DataDir + '{}/{}'.format(room, genderFlag)

    fullSaveDir = save_dir + '{}/{}/{}/{}/'.format(MatchMode, app, room, genderFlag)
    if not os.path.exists(fullSaveDir):
        os.makedirs(fullSaveDir)

    for azi1_i in range(len(Azimuth_array)-1):
        for azi2_i in range(azi1_i+1,len(Azimuth_array)):
            print('\nThe positions are {} and {} \n'.format(Azimuth_array[azi1_i], Azimuth_array[azi2_i]))
            for speaker1_i in range(SpeakerN-1):
                for speaker2_i in range(speaker1_i+1,SpeakerN):
                    print('\nThe speakers are {} and {} \n'.format(speaker1_i+1, speaker2_i+1))
                    for sequence in range(10):
                        fileName = '{}_Azi{}_Azi{}_p{}_p{}_{}'.format(genderFlag, Azimuth_array[azi1_i],
                                           Azimuth_array[azi2_i], speaker1_i+1, speaker2_i+1, sequence+1)
                        print(fileName)
                        mat = sio.loadmat(dirName+'/'+fileName)

                        # First estimate the delay of the two sources and then map it to IPD_mean_est
                        (peak_tau_array, IPD_mean_est, mixAngle_LR, mixLogPower) = tauInit(copy.deepcopy(mat), fileName)
                        IPD_mean_est_orig = IPD_mean_est.copy()
                        print('\nInitialised taus are {} and {} \n'.format(peak_tau_array[0], peak_tau_array[1]))

                        for iter in range(3):
                            # aaa =copy.deepcopy(mat)
                            # s1LogPower = mat['s1LogPower']
                            # s1LogPowerCopy = aaa['s1LogPower']
                            # s1LogPowerCopy[0,0] = 100
                            (X_train, groundtruth, mixAngle_L) = extractInputFeature(copy.deepcopy(mat), fileName, idealFlag=False, IPD_mean_est = IPD_mean_est)
                            # (X_train_new, groundtruth_new) = formatData(X_train, groundtruth, dg)
                            if MLPMode:
                                X_train_new = np.reshape(X_train,
                                                         (X_train.shape[0], X_train.shape[1] * X_train.shape[2]),
                                                         order="F")
                                groundtruth_new = groundtruth
                            else:
                                (X_train_new, groundtruth_new) = formatData(X_train, groundtruth, dg)

                            # plt.figure(8).suptitle('source1, source2')
                            # ax1 = plt.subplot(211)
                            # im1 = ax1.pcolor(groundtruth[:,0:255].transpose())
                            # plt.colorbar(im1)
                            # ax2 = plt.subplot(212, sharex=ax1)
                            # im2 = ax2.pcolor(groundtruth[:,255:510].transpose())
                            # plt.colorbar(im2)

                            # Now apply the SS algorithm
                            output = model.predict(X_train_new, verbose=1)

                            if MLPMode:
                                temp = X_train[:, 0:255, 5]
                                aa = np.minimum(output[:, 0:255], temp)
                                output[:, 0:255] = aa
                                aa = np.minimum(output[:, 255:510], temp)
                                output[:, 255:510] = aa

                                tmp = np.reshape(output, (output.shape[0], 255, 2), order='F')
                                tmp = np.transpose(tmp, (2, 1, 0))
                                loss1 = np.mean((output[:, 0:255] - groundtruth[:, 0:255]) ** 2)
                                loss2 = np.mean((output[:, 255:510] - groundtruth[:, 255:510]) ** 2)
                                print('\nThe losses are {} and {} \n'.format(loss1, loss2))
                                # showData(groundtruth.transpose(), X_train[:, 0:255, 5].squeeze().transpose(), tmp)

                                # change the output shape to the same as the hybrid DNN structure, i.e. (T 2 255 1)
                                outputCopy = np.reshape(output, (output.shape[0], 255, 2, 1), order='F')
                                outputCopy2 = np.transpose(outputCopy, (0, 2, 1, 3))
                                (peak_tau_array, IPD_mean_est) = refineIPDmean(mixAngle_LR, mixLogPower, outputCopy2, peak_tau_array, IPD_mean_est)

                            else:
                                temp = X_train[:, 0:255, 5]
                                aa = np.minimum(output[:, 0, :, 0], temp)
                                output[:, 0, :, 0] = aa
                                aa = np.minimum(output[:, 1, :, 0], temp)
                                output[:, 1, :, 0] = aa

                                tmp = np.transpose(output.squeeze(), (1, 2, 0))
                                loss1 = np.mean((tmp[0, :, :] - groundtruth[:, 0:255].transpose()) ** 2)
                                loss2 = np.mean((tmp[1, :, :] - groundtruth[:, 255:510].transpose()) ** 2)
                                print('\nThe losses are {} and {} \n'.format(loss1, loss2))
                                # showData(groundtruth.transpose(), X_train[:, 0:255, 5].squeeze().transpose(), tmp)

                                (peak_tau_array, IPD_mean_est) = refineIPDmean(mixAngle_LR, mixLogPower, output, peak_tau_array, IPD_mean_est)



                        save_mix_name = fileName+'_'+app+'_mix.wav'
                        save_est_name1 = fileName+'_'+app+'_est1.wav'
                        save_est_name2 = fileName+'_'+app+'_est2.wav'

                        # fullSaveDir = '/user/cvsspstf/ql0002/Desktop/'
                        # save_mix_name = 'mix.wav'
                        # save_est_name1 = 'est1.wav'
                        # save_est_name2 = 'est2.wav'

                        recoverObj = RecoverData(params_dir, 0.5, 16000)
                        recoverObj.recover_from_spectrum(X_train[:, 0:255, 5].squeeze(), mixAngle_L.transpose(),
                                                         fullSaveDir + save_mix_name)
                        if MLPMode:
                            recoverObj.recover_from_spectrum(output[:, 0:255], mixAngle_L.transpose(),
                                                             fullSaveDir + save_est_name1)
                            recoverObj.recover_from_spectrum(output[:, 255:510], mixAngle_L.transpose(),
                                                             fullSaveDir + save_est_name2)
                        else:
                            recoverObj.recover_from_spectrum(output[:, 0, :, 0].squeeze(), mixAngle_L.transpose(),
                                                             fullSaveDir + save_est_name1)
                            recoverObj.recover_from_spectrum(output[:, 1, :, 0].squeeze(), mixAngle_L.transpose(),
                                                             fullSaveDir + save_est_name2)
