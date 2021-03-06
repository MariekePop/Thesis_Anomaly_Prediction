""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the Keras (TensorFlow) backend
The basic idea is to predict anomalies in a time-series.
"""
import time
begin = time.time()
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import pickle

np.random.seed(1234)

# Global hyper-parameters
sequence_length = 7
batch_size = 1
dropout = 0


df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\Dataset\jips_dmch_uur_nogaps.csv')
threshold_df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\new_labels_with_thresholds.csv')

# center data around zero (same transformation for the dynamic threshold)
def z_norm(result, threshold_input, threshold_result):
    result_mean = result.mean()
    result_std = result.std()
    result = result - result_mean
    result /= result_std
    threshold_input = threshold_input - result_mean
    append = threshold_input / result_std
    threshold_result.append(append)
    try:
        threshold_result[1].tolist()
        print(len(threshold_result[1]))
        threshold_result = threshold_result[1]
    except:
        print("z_norm")
    return result, result_mean, threshold_result

# split the data into training and testset
def get_split_prep_data(train_start, train_end, test_start, test_end, threshold_result, threshold_input, date_time_data_df):
    data = df['Count']
    threshold_data = threshold_df['Threshold'].values.tolist()
    date_time_data = df['DateTime'].values.tolist()
    print("Length of Data", len(data))

    # train data
    print("Creating train data...")

    result = []
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
        threshold_input.append(threshold_data[index: index + sequence_length])
        date_time_data_df.append(date_time_data[index: index + sequence_length])
    result = np.array(result)
    result, result_mean, threshold_result = z_norm(result, threshold_input, threshold_result)

    print("Mean of train data : ", result_mean)
    print("Train data shape  : ", result.shape)

    train = result[train_start:train_end, :]
    # shuffle in-place
    np.random.shuffle(train)
    # add sequences to X_train
    X_train = train[:, :-1]
    # add target values to y_train
    y_train = train[:, -1]

    # test data
    print("Creating test data...")

    result = []
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
        threshold_input.append(threshold_data[index: index + sequence_length])
        date_time_data_df.append(date_time_data[index: index + sequence_length])
    result = np.array(result)
    result, result_mean, threshold_result = z_norm(result, threshold_input, threshold_result)

    print("Mean of test data : ", result_mean)
    print("Test data shape  : ", result.shape)

    threshold_result = threshold_result[:, -1].tolist()
    date_time_data_df = np.array(date_time_data_df)[:, -1].tolist()
    # add sequences to X_test
    X_test = result[:, :-1]
    # add target values to y_test
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))
    print("Shape y_test", np.shape(y_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, threshold_result, threshold_input, date_time_data_df

# Simple RNN model
def build_model(input_shape):
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(SimpleRNN(units=layers['hidden1'], input_shape=input_shape, input_dim=layers['input'], return_sequences=True))
    model.add(Dropout(dropout))
    model.add(SimpleRNN(units=layers['hidden2'], return_sequences=True))
    model.add(Dropout(dropout))
    model.add(SimpleRNN(units=layers['hidden3'])) 
    model.add(Dropout(dropout))
    model.add(Dense(units=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    print("model: ", model)
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()
    threshold_input = []
    threshold_result = []
    date_time_data_df = []

    if data is None:
        print('Loading data... ')
        # split data into training and testset
        X_train, y_train, X_test, y_test, threshold_result, threshold_input, date_time_data_df = get_split_prep_data(0, 20000, 20001, 25000, threshold_result, threshold_input, date_time_data_df)
    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')
    input_shape = X_train.shape[1:]

    # Build the model
    if model is None:
        model = build_model(input_shape)

    try:
        print("Training...")
        # train the model
        model.fit(
                X_train, y_train,
                batch_size=batch_size, validation_split=0.1)

        # save the model
        save_path = './Simple_RNNmodel.h5'
        model.save(save_path)

    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0
    
    # store values in files
    df2 = pd.DataFrame({
        'New_threshold': threshold_result,
    })
    df2.to_csv('tryout_Simple_RNN_thr2.csv')

    y_test_df = pd.DataFrame({
        'y_test': y_test
    })
    y_test_df.to_csv('y_test.csv')

    with open("X_test.bin", "wb") as output:
        pickle.dump(X_test, output)

    print('Training duration (s) : ', time.time() - global_start_time)

    return model, y_test


run_network()
print("Total time : ", time.time() - begin)