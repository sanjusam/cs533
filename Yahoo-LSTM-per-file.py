import sys
import os
import time
import argparse
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

def read_data_with_labels(file, timeVariantColumns, labelColumnNum):
    df = pd.read_csv(file)
    data = df.values.astype('float64')
    tsData = df[timeVariantColumns].values.astype('float64')
    labels = data[:, labelColumnNum].reshape((-1,1))
    tsDataWithLabels = np.hstack((tsData, labels))
    return tsDataWithLabels, data

def scale(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data)
    return scaler, scaler.transform(data)

def split_data_set(dataset, split=0.67):
    train_size = int(len(dataset) * split)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test

# input expected to be a 2D array with last column being label
# Returns looked back X (n_samples, n_steps, n_features) and Y (n_samples, 2); 
# last column in looked back Y data returned is label
# Only one step ahead prediction setting is expected.

def look_back_and_create_dataset(tsDataWithLabels, look_back = 1):
    lookbackTsDataX = [] 
    lookbackTsDataYAndLabel = []
    for i in range(look_back, len(tsDataWithLabels)):
        a = tsDataWithLabels[i-look_back:i, :-1]
        lookbackTsDataX.append(a)
        lookbackTsDataYAndLabel.append(tsDataWithLabels[i])
    return np.array(lookbackTsDataX), np.array(lookbackTsDataYAndLabel)

def get_train_validation(Xtrain, Ytrain, validation_ratio=0.1):
    validation_size = int(len(Xtrain) * validation_ratio)
    Xtrain, Xvalid = Xtrain[validation_size:], Xtrain[:validation_size]
    Ytrain, Yvalid = Ytrain[validation_size:], Ytrain[:validation_size]
    return Xtrain, Ytrain, Xvalid, Yvalid

def get_deviations(model, X, Y):
    deviations = np.absolute(Y - model.predict(X))
    print("Deviation Min {}, Max {}".format(np.amin(deviations, axis=0), np.amax(deviations, axis=0)))    
    return deviations

def get_records_above_deviation_pctile(model, X, Y, pctile=95):
    deviations = get_deviations(model, X, Y)
    pctileDeviationValue = np.percentile(deviations, q=pctile, axis=0)
    print("Deviation {}th pctile {}".format(pctile, pctileDeviationValue ))
    labels = (deviations > pctileDeviationValue).astype('int')
    print("Deviation > {}th pctile is_anomaly labels in data {}".format(pctile, np.unique(labels, return_counts = True)))
    return labels

def get_classification_metrics(actual, predicted):
    return confusion_matrix(actual, predicted), precision_score(actual, predicted), \
    recall_score(actual, predicted), f1_score(actual, predicted)

# Note here the slight change in how we stack the hidden LSTM layers - special for the last LSTM layer.
def baseline_model(input_shape, learning_rate):
    def build_model(input_shape=input_shape, n_hidden = 1, n_units = 50, learning_rate = learning_rate):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden - 1):
            # return sequence = true for all layers except last layer
            model.add(keras.layers.LSTM(n_units, return_sequences = True, activation = 'relu'))
        model.add(keras.layers.LSTM(n_units, activation = 'relu'))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    return build_model

def plot_actuals_vs_predictions(Y, YtrainPredicted, YtestPredicted, look_back):
    trainPredictPlot = np.empty_like(Y)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[:len(YtrainPredicted), :] = YtrainPredicted
    
    testPredictPlot = np.empty_like(Y)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(YtrainPredicted):len(Y), :] = YtestPredicted    

    #Now Plot
    plt.figure(figsize=(40,10))
    plt.plot(testPredictPlot, label ='Test Prediction')
    plt.plot(trainPredictPlot, label ='Train Prediction')
    plt.plot(Y, label = 'Actual')
    plt.legend(("Test Prediction", "Train Prediction", "Actual"), loc=3)
    plt.show()
    
"""
The `perform_training_on_benchmark_directory` is a wrapper fucntion which calls data_read, look_back, split methords  

Decription for each parameters

    benchmark_dir             - The full path to a folder where the data resides. This will be the path to a A* Benchmark folders.
    extension_pattern         - File name extension pattern for files of this folder.
    timeVariantColumns        – The time variant column specifies a list of values that changes in time.  In case of Yahoo! It’s the “value” column.
    labelColumnNum            – Specifies the column number which denotes if that record is anomaly or not.  The case of A1Benchmark and A2Benchmark folder the field is marked as “is_anomaly” whereas in case of A3Benchmark and A4Benchmark its “anomaly”
    param_distribs             - Distribution of parameters to search for best model using randomized search.
    files_to_process  - Specifies the number of files that needs to be processed per directory.  
    plot_graph                          – Specifies to plot graph or not for each file.    
    validation_ratio    - What fraction of Xtrain, Ytrain to be used for early stopping validation.
    early_stop_patience - Stop optimizing after how many successive epochs without further loss reduction
    epochs              - Total number of epochs to try and optimize loss.
    batch_size          - Size of batches in each epoch
    n_iter              - number of iterations
    cv                  - cv
    verbosity
    save_model 
"""
    
# Do it on each benchmark directory files  
def perform_training_on_benchmark_directory(benchmark_dir, extension, timeVariantColumns, 
                                            labelColumnNum, param_distribs, files_to_process = 'ALL', plot_graph = 1,
                                            validation_ratio = 0.1, early_stop_patience = 5, epochs = 25, batch_size = 32,
                                            n_iter = 1, cv = 5, verbosity = 0, save_model=0, file_name_preferred = None):
    pctile = 99.5
    split = 0.8
    look_back = 24
    learning_rate = 0.001

    Benchmark_dir  = YAHOO_DS + os.path.sep + benchmark_dir + os.path.sep 
    benchmark_files = glob.glob(Benchmark_dir + extension, recursive=True)
    
    if files_to_process == 'ALL' :
        num_files_to_process = len(benchmark_files)
    else :
        num_files_to_process = int(files_to_process)
    

    resultsMap={} # results from this folder    
    files_to_walk = []
    if file_name_preferred == None :
        files_to_walk = benchmark_files[:num_files_to_process]
    else :
        files_to_walk.append(Benchmark_dir + file_name_preferred)
        num_files_to_process = 1;

    print('Processing {} files in folder {}'.format(num_files_to_process, benchmark_dir)) 
    #for file_name in benchmark_files[:num_files_to_process]:
    for file_name in files_to_walk:
        keras.backend.clear_session()
        print('File Name : ', file_name)
        
        # read data        
        tsDataWithLabels, data = read_data_with_labels(file_name, timeVariantColumns, labelColumnNum)
        print("Shapes: time variant data with labels {}, full data {}".format(tsDataWithLabels.shape, data.shape))
        
        # scale data
        scaler, tsDataScaled = scale(tsDataWithLabels) 
        
        # Get look back data in the 3D array shape
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

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=early_stop_patience, restore_best_weights=True, monitor='val_loss', min_delta=0.0003)

        rnd_search_cv = RandomizedSearchCV(regressor, param_distribs, n_iter = n_iter, cv = cv, verbose = verbosity)

        start_millis = current_time_millis()
        rnd_search_cv.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xvalid, Yvalid), 
                          callbacks=[early_stopping_cb], verbose=verbosity)


        end_millis = current_time_millis()

        model = rnd_search_cv.best_estimator_.model
        print("Best parameters {} best score {}:".format(rnd_search_cv.best_params_, -rnd_search_cv.best_score_))

        trainMSE = model.evaluate(Xtrain_full, Ytrain_full, verbose = verbosity)
        print("Train Score: {0:.5f} MSE {1:.5f} RMSE".format(trainMSE, np.sqrt(trainMSE)))
        testMSE = model.evaluate(Xtest, Ytest, verbose = verbosity)
        print("Test Score: {0:.5f} MSE {1:.5f} RMSE".format(testMSE, np.sqrt(testMSE)))
        
                
        
        # get deviations for whole dataset and id records with deviations > pctile threshold and asign an is_anomaly label
        predictedLabels = get_records_above_deviation_pctile(model, lookbackX, lookbackY[:, :-1], pctile)

        # actual is_anomaly labels in dataset
        actualLabels = (data[look_back:, labelColumnNum] != 0.0).astype('int')    
        print("Actual is_anomaly labels in data", np.unique(actualLabels, return_counts = True))

        # Compare calculated labels and actual labels to find confusion matrix, precision, recall, and F1
        conf_matrix, prec, recall, f1 = get_classification_metrics(actualLabels, predictedLabels)
        print("Confusion matrix \n{0}\nprecision {1:.5f}, recall {2:.5f}, f1 {3:.5f}".format(conf_matrix, prec, recall, f1))
        print("Time to train: {} ms".format(end_millis - start_millis))
        resultsMap[file_name] = {'traintime' : (end_millis - start_millis), 'model' : model, 
                                'best params' : rnd_search_cv.best_params_, 'best score' : -rnd_search_cv.best_score_,
                                'train MSE' : trainMSE, 'test MSE' : testMSE,
                                'precision' : prec, 'recall' : recall, 'f1' : f1, 'confusion_matrix' : conf_matrix}

        if save_model == 1 :
            modeldir = 'models' + '/' + benchmark_dir
            if not os.path.exists(modeldir):
                os.makedirs(modeldir) 
            modelpath = modeldir + '/' + file_name.split('/')[-1] + '.h5'
            print("Saving model", modelpath )
            model.save(modelpath)
         
        
    return resultsMap  

def print_summary_for_benchmark_folder(resultsMap, benchmark_folder):
    precisions=[]
    recalls=[]
    f1s=[]
    times = []
    for v in resultsMap.values():
        precisions.append(v['precision'])
        recalls.append(v['recall'])
        f1s.append(v['f1'])
        times.append(v['traintime'])
    avg_prec = np.average(np.array(precisions))
    avg_recall = np.average(np.array(recalls))
    avg_f1 = np.average(np.array(f1s))
    avg_time = np.average(np.array(times))
    print(benchmark_folder, ": Avg precision {0:.5f} recall {1:.5f} f1 {2:.5f} time to train {3:.2f} ms".
          format(avg_prec, avg_recall, avg_f1, avg_time)) 

###################### start main ##################

parser = argparse.ArgumentParser()
parser.add_argument('--num_threads_inter', type=int, default=4, required=False)
parser.add_argument('-f', '--folder', type=str, default=None, required=True, action='append')
parser.add_argument('--num_files', type=str, default='ALL', required=False)
parser.add_argument('--n_iter', type=int, default=1, required=False)
parser.add_argument('--epochs', type=int, default=25, required=False)
parser.add_argument('--batch_size', type=int, default=32, required=False)
parser.add_argument('--patience', type=int, default=5, required=False)
parser.add_argument('--cv', type=int, default=5, required=False)
parser.add_argument('--save', type=int, default=0, required=False)
parser.add_argument('--verbose', type=int, default=0, required=False)
parser.add_argument('--file_name', type=str, default=None, required=False)

args = parser.parse_args()

folders=args.folder
files_to_process=args.num_files
n_iter = args.n_iter
cv = args.cv
batch_size = args.batch_size
epochs = args.epochs
early_stop_patience = args.patience
verbosity = args.verbose
save_model=args.save
file_name_preferred=args.file_name

tf.config.threading.set_inter_op_parallelism_threads(args.num_threads_inter)
    
print('python version', sys.version_info)
print('tf version', tf.__version__, 'keras version', keras.__version__)

current_time_millis = lambda: int(round(time.time() * 1000))

YAHOO_DS="../../Stochastic-Methods/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0"
DIRS_FILE_EXTENSIONS = {'A1Benchmark' : "*.csv", \
                        'A2Benchmark' : "*.csv", \
                        'A3Benchmark' : "*TS*.csv", \
                        'A4Benchmark' : "*TS*.csv" }


#param_distribs = {
#    "n_hidden": np.arange(1, 2).tolist(), # upto 3 hidden layers
#    #"n_units": np.arange(24, 48 ).tolist() # 72 - 97 hidden layer units/neurons
#    "n_units": [24]  # 72 - 97 hidden layer units/neurons
#}
param_distribs = {
    "n_hidden": np.arange(1, 3).tolist(), # upto 3 hidden layers
    #"n_units": [24, 48, 72, 96]  # 72 - 97 hidden layer units/neurons
    "n_units": [48, 72, 96]  # 72 - 97 hidden layer units/neurons
}

directoryResultsMap = {}
start_millis = current_time_millis()

for folder in folders:
    extension = DIRS_FILE_EXTENSIONS[folder]
    timeVariantColumns = ['value']
    labelColumnNum = 2
    resultsMap = perform_training_on_benchmark_directory(folder, extension, timeVariantColumns, 
                                            labelColumnNum, param_distribs, files_to_process, plot_graph = 1, 
                                            early_stop_patience = early_stop_patience, epochs = epochs, batch_size = batch_size,
                                            n_iter = n_iter, cv = cv, verbosity = verbosity, save_model=save_model, file_name_preferred=file_name_preferred)
    directoryResultsMap[folder] = resultsMap

end_millis = current_time_millis()
print("Total Time to proces all the selected files : {} ms".format(end_millis - start_millis))


#finally Print summary for each directory
for directory in directoryResultsMap.keys() :
    print_summary_for_benchmark_folder(directoryResultsMap[directory], directory)
