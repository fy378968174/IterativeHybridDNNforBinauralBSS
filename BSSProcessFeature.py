
from keras import layers
from keras import models
from keras.utils import plot_model
from keras.optimizers import SGD

import numpy as np
from STFT import istft
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os


from DataGeneration import dataGenBig
from Others import formatData
from GenerateModels import generateModel

# import theano
# print('theano: ', theano.__version__)
# import sys
# print(sys.version)
# import keras
# print('keras: ', keras.__version__)





model = generateModel()
# print(model.summary())
# plot_model(model, to_file='model.png')



# # =========continue to train the model
# optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=optim)
# model.load_weights('BSSmodelProcessFeature27Sept')
# dg = dataGenBig(False,87654321)
# fileName = "ProcessFeature0001.txt"


# =========train a new model
optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=optim)
dg = dataGenBig(False,123456789)
fileName = "ProcessFeature0005.txt"




# The parameters for training the model
dg.TrainDataParams()


saveModelName = "BSSmodelProcessFeature"
# if os.path.exists(fileName):
#     os.remove(fileName)
wr = open(fileName, "w")

for epoch in range(100):
    stri = '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Epoch is {}\n'.format(epoch)
    print(stri)
    wr.write(stri)
    wr.flush()
    for index in range(int(dg.trainNum/dg.BATCH_SIZE_Train)):
        if index % 10 == 0:
            model.save_weights(saveModelName, True)

        if index % 5 == 0:
            (X_valid, y_valid) = dg.batchCallableValidation()
            (X_valid_new, y_valid_new) = formatData(X_valid, y_valid, dg)
            loss = model.evaluate(X_valid_new, y_valid_new, batch_size=X_valid_new.shape[0], verbose=1)
            stri = "*****************batch %d valid_loss : %f\n" % (index, loss)
            print(stri)
            wr.write(stri)
            wr.flush()

        (X_train, y_train) = dg.batchCallableTrain()
        (X_train_new,y_train_new) = formatData(X_train, y_train, dg)

        loss = model.train_on_batch(X_train_new, y_train_new)
        stri = "==================batch %d train_loss : %f\n" % (index, loss)
        print(stri)
        wr.write(stri)
        wr.flush()

wr.close()



