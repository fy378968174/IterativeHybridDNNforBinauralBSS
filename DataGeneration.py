import numpy as np
import scipy.io as sio
import os
from random import shuffle
import random
import time
import math
import re
import matplotlib.pyplot as plt

class dataGenBig:
    def __init__(self,IBMmode=False, seedNum = 123456789):
        self.IBMmode = IBMmode
        mask_threshold = 0 #-10 # 10log10(sig^2)>10log10(noi^2)+mask_threshold
        self.real_threshold = mask_threshold / 10 * math.log(10)

        # For each sequence, randomly chose 8 feature blocks
        self.NeachSeq = 8
        self.batchLen = 11 # each sequence contains 100 frames
        self.halfBatchLen = int((self.batchLen - 1) / 2)

        self.BATCH_SIZE_Train = 128 #32 # 128 #4096  # mini-batch size
        self.batchSeqN_Train = int(round(self.BATCH_SIZE_Train / self.NeachSeq))
        self.BATCH_SIZE_Train = self.batchSeqN_Train * self.NeachSeq

        self.BATCH_SIZE_Valid = 512  # 32 # 128 #4096  # mini-batch size
        self.batchSeqN_Valid = int(round(self.BATCH_SIZE_Valid / self.NeachSeq))
        self.BATCH_SIZE_Valid = self.batchSeqN_Valid * self.NeachSeq

        self.train_i = 0
        self.valid_i = 0


        # The mean and variance of the features to normalise the input and output feature vector
        mat2 = sio.loadmat('/ATTENTION_CHANGE_DIR_HERE/Params/RoomCmyNormParamsMixture.mat')
        sig_std = mat2['sig_std']  # Dimx1
        sig_mean = mat2['sig_mean']  # Dimx1
        self.sig_std = sig_std.reshape(sig_std.size, )
        self.sig_mean = sig_mean.reshape(sig_std.size, )

        # the IPD parameters extracted from clean signals
        # The mean and variance of the features to normalise the input and output feature vector
        mat1 = sio.loadmat('/ATTENTION_CHANGE_DIR_HERE/Params/RoomCGMMParams.mat')
        IPD_mean = mat1['IPD_mean']
        self.IPD_mean = np.asarray(IPD_mean).transpose()
        # IPD_var = mat1['IPD_var']
        # self.IPD_var = np.asarray(IPD_var).transpose()
        # self.IPD_var2 = -2*self.IPD_var
        # self.mlog2pisigma = -0.5*np.log(2*np.pi*self.IPD_var)
        Azimuth_array = mat1['Azimuth_array']
        self.Azimuth_array = Azimuth_array.reshape(Azimuth_array.size, ).tolist()


        self.halfNFFT = self.sig_mean.size
        self.halfNFFTtrim = self.halfNFFT-2
        self.Dim_out = self.halfNFFTtrim*2
        self.Dim_in = self.halfNFFTtrim*3

        self.miniTrainN = 0
        self.miniValidN = 0
        self.seedNum = seedNum




    def TrainDataParams(self, Train_DIR = '/ATTENTION_CHANGE_DIR_HERE/DLTrainingData/'):
        self.Train_DIR = Train_DIR

        TotalFilenameList = list()
        for room in ['RoomC']:
            for gender_level in ['MM','MF','FF']:
                dirName = self.Train_DIR + '{}/{}'.format(room,gender_level)
                filenameList = os.listdir(dirName)
                TotalFilenameList += filenameList

        TotalSeqNum = len(TotalFilenameList)
        # The training data will be divided to two part, train and valid
        trainNum = int(round(TotalSeqNum*0.8))
        validNum = TotalSeqNum - trainNum


        print('The total number of training data is ' + str(TotalSeqNum) + ', of which ' +
              str(trainNum) + ' are used for training and ' + str(validNum) + ' for validation.')

        # shuffle the list
        random.seed(999999999)  # Note here, the train and valid should not be muddled up at different training sessions
        shuffle(TotalFilenameList)

        self.trainList = TotalFilenameList[0:trainNum]
        self.validList = TotalFilenameList[trainNum:]

        random.seed(self.seedNum)
        self.validNum = len(self.validList)
        self.trainNum = len(self.trainList)

        print('shuffle the train and valid data')
        shuffle(self.trainList)
        shuffle(self.validList)

        print(self.trainList[0], self.validList[0])

    def batchCallableTrain(self):  # This callable should have NO input
        # global train_i, trainList

        # Be careful about the data type!
        BatchDataIn = np.zeros((self.BATCH_SIZE_Train, self.Dim_in, self.batchLen), dtype='f')
        BatchDataOut = np.zeros((self.BATCH_SIZE_Train, self.Dim_out), dtype='f')

        self.miniTrainN += 1
        print('Now collect a mini batch for training----{}'.format(str(self.miniTrainN)))

        i = 0
        N = 8
        while i < self.batchSeqN_Train:

            failN = 0
            while True:
                try:
                    if self.train_i >= self.trainNum:
                        self.train_i = 0
                        print('Run through all the training data, and shuffle the data again!')
                        shuffle(self.trainList)
                        print(self.trainList[0])

                    filename = self.trainList[self.train_i]
                    subFolder = filename.split('_')[0]
                    folderName = self.Train_DIR + 'RoomC/' + subFolder
                    temp = folderName + '/' + filename
                    mat = sio.loadmat(temp)
                    # NotLoadFlag = False
                    # we want to extract the same batchLen frames for each sequence,
                    s1LogPower = mat['s1LogPower']
                    s1LogPower = np.asarray(s1LogPower)
                    if s1LogPower.shape[1] >= self.batchLen:
                        break
                    else:
                        self.train_i += 1
                except:
                    failN += 1
                    time.sleep(failN)  # This is very important, otherwise, the failure persists
                    # if (failN > 1) & (failN % 5 == 0):
                    #     global sio
                    #     import scipy.io as sio
                    print('Failed {} times, Try to load the next sequence ---- {} '.format(failN, self.train_i))
                    self.train_i += 1

            (currentBatchDataIn, currentBatchDataOut, _) = self.processOneBatchFromMat(mat, filename)
            BatchDataIn[i * N: (i + 1) * N, :, :] = currentBatchDataIn
            BatchDataOut[i * N: (i + 1) * N, :] = currentBatchDataOut

            self.train_i += 1
            i += 1
        # print(self.valid_i)

        print('Training batch data {} collected'.format(self.miniTrainN))
        return [BatchDataIn, BatchDataOut]


    def processOneBatchFromMat(self, mat, filename, batchFlag=True):

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

        # if batchFlag:
        #     # randomly choose the starting index in each sequence
        #     chooseIndex = random.randint(0, s1LogPower.shape[1] - self.batchLen)
        #     # print(chooseIndex)
        #
        #     s1LogPower = s1LogPower[:, chooseIndex:chooseIndex+self.batchLen]
        #     s2LogPower = s2LogPower[:, chooseIndex:chooseIndex+self.batchLen]
        #     mixAngle_L = mixAngle_L[:, chooseIndex:chooseIndex+self.batchLen]
        #     mixAngle_R = mixAngle_R[:, chooseIndex:chooseIndex+self.batchLen]
        #     mixLogPower = mixLogPower[:, chooseIndex:chooseIndex+self.batchLen]

        mixAngle_LR = mixAngle_L[1:-1, :] - mixAngle_R[1:-1, :]
        # map the angle different between [-pi pi]
        mixAngle_LR[mixAngle_LR > np.pi] = mixAngle_LR[mixAngle_LR > np.pi] - 2 * np.pi
        mixAngle_LR[mixAngle_LR < -np.pi] = mixAngle_LR[mixAngle_LR < -np.pi] + 2 * np.pi

        Nums = list(map(int, re.findall('[+-]?\d+', filename)))
        azi1_index = self.Azimuth_array.index(Nums[0])  # np.where(self.Azimuth_array == Nums[0])[0][0]
        azi2_index = self.Azimuth_array.index(Nums[1])  # angel of the second speaker
        # shiftAng1 = [mixAngle_LR[:,k]-self.IPD_mean[azi1_index] for k in range(mixLogPower.shape[1])]
        shiftAng1 = mixAngle_LR.copy()
        shiftAng2 = mixAngle_LR.copy()
        for k in range(mixLogPower.shape[1]):
            shiftAng1[:, k] -= self.IPD_mean[azi1_index]
            shiftAng2[:, k] -= self.IPD_mean[azi2_index]

        shiftAng1[shiftAng1 > np.pi] = shiftAng1[shiftAng1 > np.pi] - 2 * np.pi
        shiftAng1[shiftAng1 < -np.pi] = shiftAng1[shiftAng1 < -np.pi] + 2 * np.pi
        shiftAng2[shiftAng2 > np.pi] = shiftAng2[shiftAng2 > np.pi] - 2 * np.pi
        shiftAng2[shiftAng2 < -np.pi] = shiftAng2[shiftAng2 < -np.pi] + 2 * np.pi

        shiftAng1 = np.power(shiftAng1, 2)
        shiftAng2 = np.power(shiftAng2, 2)

        shiftAng1 = np.exp(-shiftAng1)
        shiftAng2 = np.exp(-shiftAng2)

        # for k in range(mixLogPower.shape[1]):
        #     shiftAng1[:, k] /= self.IPD_var2[azi1_index]
        #     shiftAng1[:, k] += self.mlog2pisigma[azi1_index]
        #     shiftAng2[:, k] /= self.IPD_var2[azi2_index]
        #     shiftAng2[:, k] += self.mlog2pisigma[azi2_index]
        #
        # shiftAng1 = np.exp(-shiftAng1)
        # shiftAng2 = np.exp(-shiftAng2)
        # inputMask = np.divide(shiftAng1, shiftAng1 + shiftAng2)

        # inputMask = shiftAng1 / (shiftAng1 + shiftAng2)
        # plt.figure(101)
        # plt.pcolor(np.abs(inputMask))
        # plt.show()



        if self.IBMmode:
            # noiLogPower = mat['noiLogPower']  # Dim*#Frames
            # noiLogPower = np.asarray(noiLogPower)
            # IBM
            # mask = sigLogPower > (noiLogPower + self.real_threshold)
            # IRM
            sigLogPower2 = np.exp(s1LogPower)
            noiLogPower2 = np.exp(s2LogPower)
            mask = np.sqrt(np.divide(sigLogPower2, sigLogPower2 + noiLogPower2))
            # normalise the data(prewhitening)
            for k in range(mixLogPower.shape[1]):
                mixLogPower[:, k] -= self.sig_mean
                mixLogPower[:, k] /= self.sig_std
        else:
            # normalise the data(prewhitening)
            for k in range(mixLogPower.shape[1]):
                s1LogPower[:, k] -= self.sig_mean
                s1LogPower[:, k] /= self.sig_std
                s2LogPower[:, k] -= self.sig_mean
                s2LogPower[:, k] /= self.sig_std
                mixLogPower[:, k] -= self.sig_mean
                mixLogPower[:, k] /= self.sig_std


        if batchFlag:
            # randomly choose 8 feature vectors in each sequence
            # chooseIndex = np.random.randint(0, mixLogPower.shape[1]-self.batchLen, 32)
            N = 8
            try:
                a = np.arange(mixLogPower.shape[1] - self.batchLen)
                shuffle(a)
                chooseIndex = a[:N]
            except:  # for short sequences that cannot yield 32 none-duplicated data
                chooseIndex = random.randint(0, mixLogPower.shape[1] - self.batchLen, N)
        else:
            N = mixLogPower.shape[1] - self.batchLen+1
            chooseIndex = np.arange(N)  # (range(0, N)


        # concatenate the feature as the input
        Index1 = (np.tile(range(0, self.batchLen), (N, 1))).transpose()
        Index2 = np.tile(chooseIndex, (self.batchLen, 1))
        Index = Index1 + Index2
        mixLogPower = mixLogPower[:, Index]

        # tmp = np.reshape(tmp, (self.Dim_in, N), order="F")
        mixLogPower = np.transpose(mixLogPower, (2, 0, 1))

        shiftAng1 = shiftAng1[:, Index]
        shiftAng1 = np.transpose(shiftAng1, (2, 0, 1))

        shiftAng2 = shiftAng2[:, Index]
        shiftAng2 = np.transpose(shiftAng2, (2, 0, 1))

        # s1LogPower = s1LogPower[:, Index]
        # s1LogPower = np.transpose(s1LogPower, (2, 0, 1))
        #
        # s2LogPower = s2LogPower[:, Index]
        # s2LogPower = np.transpose(s2LogPower, (2, 0, 1))



        currentBatchDataIn = np.concatenate((mixLogPower[:, 1:-1, :], shiftAng1, shiftAng2), axis=1)
        if self.IBMmode:
            currentBatchDataOut = mask.transpose()
        else:
            currentBatchDataOut = (np.concatenate((s1LogPower[1:-1, chooseIndex+self.halfBatchLen], s2LogPower[1:-1, chooseIndex+self.halfBatchLen]), axis=0)).transpose()

        return (currentBatchDataIn, currentBatchDataOut, chooseIndex)


    def batchCallableValidation(self):  # This callable should have NO input
        # global train_i, trainList

        # Be careful about the data type!
        BatchDataIn = np.zeros((self.BATCH_SIZE_Valid, self.Dim_in, self.batchLen), dtype='f')
        BatchDataOut = np.zeros((self.BATCH_SIZE_Valid, self.Dim_out), dtype='f')


        self.miniValidN += 1
        print('Now collect a mini batch for validation----{}'.format(str(self.miniValidN)))

        i = 0
        N = 8
        while i < self.batchSeqN_Valid:

            failN = 0
            while True:
                try:
                    if self.valid_i >= self.validNum:
                        self.valid_i = 0
                        print('Run through all the validation data, and shuffle the data again!')
                        shuffle(self.validList)
                        print(self.validList[0])

                    filename = self.validList[self.valid_i]
                    subFolder = filename.split('_')[0]
                    folderName = self.Train_DIR + 'RoomC/'+subFolder
                    temp = folderName + '/' + filename
                    mat = sio.loadmat(temp)
                    # NotLoadFlag = False
                    # we want to extract the same batchLen frames for each sequence,
                    s1LogPower = mat['s1LogPower']
                    s1LogPower = np.asarray(s1LogPower)
                    if s1LogPower.shape[1]>=self.batchLen:
                        break
                    else:
                        self.valid_i += 1
                except:
                    failN += 1
                    time.sleep(failN)  # This is very important, otherwise, the failure persists
                    # if (failN > 1) & (failN % 5 == 0):
                    #     global sio
                    #     import scipy.io as sio
                    print('Failed {} times, Try to load the next sequence ---- {} '.format(failN, self.valid_i))
                    self.valid_i += 1

            (currentBatchDataIn, currentBatchDataOut, _) = self.processOneBatchFromMat(mat, filename)
            BatchDataIn[i * N: (i + 1) * N, :, :] = currentBatchDataIn
            BatchDataOut[i * N: (i + 1) * N, :] = currentBatchDataOut

            self.valid_i += 1
            i += 1
        # print(self.valid_i)
        print('Validation batch data {} collected'.format(self.miniValidN))
        return [BatchDataIn, BatchDataOut]


    def testDataGenerationFromDir(self,data_with_dir, batchFlag=True):
        # Be careful about the data type!

        print('Now collect data sequence from directory {}'.format(data_with_dir))
        mat = sio.loadmat(data_with_dir)

        filename = data_with_dir.split('/')[-1]
        (BatchDataIn, BatchDataOut, chooseIndex) = self.processOneBatchFromMat(mat, filename, batchFlag)

        mixAngle_L = mat['mixAngle_L']
        mixAngle_L = np.asarray(mixAngle_L)
        mixAngle_L = mixAngle_L[1:-1, chooseIndex + self.halfBatchLen]

        return [BatchDataIn, mixAngle_L, BatchDataOut]


