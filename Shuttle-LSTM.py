import sys
import os
import time
import glob
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

current_time_millis = lambda: int(round(time.time() * 1000))

def label_outliers(nasa_df_row):
    if nasa_df_row['class'] == 1 :
        return 0
    else :
        return 1
    
def cleanup() :
    colnames =['time', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'class']
    train_df = pd.read_csv("../Stochastic-Methods/data/nasa/shuttle.trn/shuttle.trn",names=colnames,sep=" ")
    test_df = pd.read_csv("../Stochastic-Methods/data/nasa/shuttle.tst",names=colnames,sep=" ")

    # merge train and test
    merged_df = pd.concat([train_df, test_df])
    # print("Unique classes {}".format(np.unique(merged_df['class'].values, return_counts=True)))

    # drop class = 4
    minus4_df = merged_df.loc[merged_df['class'] != 4]
    # print("Frame after dropping 4 \n{}".format(minus4_df))
    # print("Unique classes after dropping 4 {}".format(np.unique(minus4_df['class'].values, return_counts=True)))

    # mark class 1 as inlier and rest as outlier
    is_anomaly_column = minus4_df.apply(lambda row: label_outliers(row), axis=1)
    labelled_df = minus4_df.assign(is_anomaly=is_anomaly_column.values)

    #print("Frame after labelling outliers \n{}".format(labelled_df))
    print("Unique classes after labelling outliers {}".format(np.unique(labelled_df['class'].values, return_counts=True)))
    print("Unique outliers after labelling outliers {}".format(np.unique(labelled_df['is_anomaly'].values, return_counts=True)))

    # sort by time

    sorted_df = labelled_df.sort_values('time')

    #print("Sorted Frame\n{}".format(sorted_df))
    
    return sorted_df

def read_data_with_labels(df, timeVariantColumns, labelColumnNum):
#     df = pd.read_csv(file)
    data = df.values.astype('float64')
    tsData = df[timeVariantColumns].values.astype('float64')
    labels = data[:, labelColumnNum].reshape((-1,1))
    tsDataWithLabels = np.hstack((tsData, labels))
    return tsDataWithLabels, data

def scale(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data)
    return scaler, scaler.transform(data)

"""
# input expected to be a 2D array with last column being label
# Returns looked back X adn Y; last column in look back Y data returned is label
# Only one step ahead prediction setting is expected.
"""

def look_back_and_create_dataset(tsDataWithLabels, look_back = 1):
    lookbackTsDataX = [] 
    lookbackTsDataYAndLabel = []
    for i in range(look_back, len(tsDataWithLabels)):
        a = tsDataWithLabels[i-look_back:i, :-1]
        lookbackTsDataX.append(a)
        lookbackTsDataYAndLabel.append(tsDataWithLabels[i])
    return np.array(lookbackTsDataX), np.array(lookbackTsDataYAndLabel)

def split_data_set(dataset, split=0.67):
    train_size = int(len(dataset) * split)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test

def get_train_validation(Xtrain, Ytrain, validation_ratio=0.1):
    validation_size = int(len(Xtrain) * validation_ratio)
    Xtrain, Xvalid = Xtrain[validation_size:], Xtrain[:validation_size]
    Ytrain, Yvalid = Ytrain[validation_size:], Ytrain[:validation_size]
    return Xtrain, Ytrain, Xvalid, Yvalid

# Note here the slight change in how we stack the hidden LSTM layers - special for the last LSTM layer.
def baseline_model(input_shape, learning_rate):
    def build_model(input_shape=input_shape, n_hidden = 1, n_units = 50, learning_rate = learning_rate):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden - 1):
            # return sequence = true for all layers except last layer
            model.add(keras.layers.LSTM(n_units, return_sequences = True, activation = 'relu'))
        model.add(keras.layers.LSTM(n_units, activation = 'relu'))
        model.add(keras.layers.Dense(input_shape[1]))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    return build_model

    

############## main #########################

split = 0.8
look_back = 24
learning_rate = 0.001
n_iter = 5
cv = 5
batch_size=32
early_stop_patience=3
epochs=20
verbosity=0
min_delta=0.0003

param_distribs = {
    "n_hidden": np.arange(1, 3).tolist(), # upto 1 hidden layers
    #"n_units": np.arange(5,6).tolist() # 5 hidden layer units/neurons
    "n_units" : [24, 48, 72, 96]
}


sorted_df = cleanup()

timeVariantColumns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
labelColumnNum = 10

# read data
tsDataWithLabels, data = read_data_with_labels(sorted_df, timeVariantColumns, labelColumnNum)
# print("Shapes: time variant data array with labels {}, full data {}".format(tsDataWithLabels.shape, data.shape))
# print("Unique outliers in full data array {}".format(np.unique(data[:, -1], return_counts=True)))
# print("Unique outliers in time variant data array with labels {}".format(np.unique(tsDataWithLabels[:, -1], 
#                                                                                    return_counts=True)))

# print(tsDataWithLabels)

# scale data
scaler, tsDataScaled = scale(tsDataWithLabels)

# Get look back data in the 3D array shape (n_samples, n_lookback_steps, n_features)
lookbackX, lookbackY = look_back_and_create_dataset(tsDataScaled, look_back=look_back)
print("Look back data shapes: lookbackX {} lookbackY {}".format(lookbackX.shape, lookbackY.shape))

 # split into train/test
Xtrain_full, Xtest = split_data_set(lookbackX, split=0.8)
Ytrain_full, Ytest = split_data_set(lookbackY[:, :-1], split=0.8)   # exclude labels     

print("Shapes: Xtrain_full {}, Ytrain_full {}, Xtest {}, Ytest {}".format(Xtrain_full.shape, Ytrain_full.shape, 
                                                                          Xtest.shape, Ytest.shape))

# split further full train set into train and validation set
Xtrain, Ytrain, Xvalid, Yvalid = get_train_validation(Xtrain_full, Ytrain_full, validation_ratio=0.1)

print("Shapes: Xtrain {}, Ytrain {}, Xvalid {}, Yvalid {}".format(Xtrain.shape, Ytrain.shape, 
                                                                  Xvalid.shape, Yvalid.shape))


input_shape = (Xtrain.shape[1], Xtrain.shape[2])
regressor = keras.wrappers.scikit_learn.KerasRegressor(build_fn = baseline_model(input_shape=input_shape, 
                                                                         learning_rate=learning_rate))

early_stopping_cb = keras.callbacks.EarlyStopping(patience=early_stop_patience, monitor='val_loss', min_delta=0.0003, 
                                                  restore_best_weights = True)

rnd_search_cv = RandomizedSearchCV(regressor, param_distribs, n_iter = n_iter, cv = cv, verbose = verbosity)

start_millis = current_time_millis()
rnd_search_cv.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xvalid, Yvalid), 
                  callbacks=[early_stopping_cb], verbose=verbosity)


end_millis = current_time_millis()

print("Time to train {}".format(end_millis -start_millis))

model = rnd_search_cv.best_estimator_.model
print("Best parameters {} best score {}:".format(rnd_search_cv.best_params_, -rnd_search_cv.best_score_))

trainMSE = model.evaluate(Xtrain_full, Ytrain_full, verbose = verbosity)
print("Train Score: {0:.5f} MSE {1:.5f} RMSE".format(trainMSE, np.sqrt(trainMSE)))
testMSE = model.evaluate(Xtest, Ytest, verbose = verbosity)
print("Test Score: {0:.5f} MSE {1:.5f} RMSE".format(testMSE, np.sqrt(testMSE)))

modeldir = 'model-shuttle-lstm'
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
modelpath = modeldir + '/' + 'shuttle-lstm.h5'
print("Saving model", modelpath )
model.save(modelpath)        
