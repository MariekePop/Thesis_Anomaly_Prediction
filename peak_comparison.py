# compare all methods on prediction vs original value on anomalies

# Transformer

import time
begin = time.time()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import pickle

df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\Dataset\jips_dmch_uur_nogaps.csv')
threshold_df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\new_labels_with_thresholds.csv')
df2 = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\tryout_Transformer_thr2.csv')
y_test_df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\y_test.csv')

# get model from file
save_path = './Transformermodel.h5'
model = keras.models.load_model(save_path)

with open("X_test.bin", "rb") as data:
    X_test = pickle.load(data)

# predict
print("Predicting...")
predicted = model.predict(X_test)
print("Reshaping predicted")
predicted = np.reshape(predicted, (predicted.size,))
predicted = predicted[1:]

# make lists for calculations result
anomaly_list = []
threshold_list = []
y_test = []

for ij in range(20007, 24989):
    anomaly_list.append(threshold_df.Anomaly[ij])

for ij in range(19993, 24975):
    threshold_list.append(df2.New_threshold[ij])

for ij in range(len(y_test_df)):
    y_test.append(y_test_df.y_test[ij])


# calculate TP, TN, FN, and FP
True_positives = 0
True_negatives = 0
False_negatives = 0
False_positives = 0

anomaly_count = 0
Tr_anomaly_predictions = []
Tr_anomaly_values = []
Tr_lower_thresh = (np.array(threshold_list)*0.65)

for original, anomaly, predict, thresh in zip(y_test, anomaly_list, predicted, Tr_lower_thresh):
    if anomaly and predict >= thresh:
        True_positives = True_positives + 1
        anomaly_count = anomaly_count+1
        Tr_anomaly_predictions.append(predict)
        Tr_anomaly_values.append(original)
    elif not anomaly and predict < thresh:
        True_negatives = True_negatives + 1
    elif anomaly and predict < thresh:
        False_negatives = False_negatives + 1
        anomaly_count = anomaly_count+1
        Tr_anomaly_predictions.append(predict)
        Tr_anomaly_values.append(original)
    else:
        False_positives  = False_positives + 1

print("True_negatives: ", True_negatives)
print("True_positives: ", True_positives)
print("False_negatives: ", False_negatives)
print("False_positives: ", False_positives)

try:
    # Calculate precision, recall, and the F1-score
    print("incorrect: ", False_negatives + False_positives)
    precision = True_positives/(True_positives + False_positives)
    print("precision: ", precision)
    recall = True_positives/(True_positives + False_negatives)
    print("recall: ", recall)
    F1_score = 2*True_positives/(2*True_positives + False_positives + False_negatives)
    print("F1_score: ", F1_score)
except:
    print("error in calculations")
try:
    Transformer_predicted = predicted
    # plt.plot(X_train_list, 'p', label='X_train')
except Exception as e:
    print("plotting exception")
    print(str(e))
print("Total time : ", time.time() - begin)


# LSTM

import time
begin = time.time()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import pickle


df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\Dataset\jips_dmch_uur_nogaps.csv')
threshold_df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\new_labels_with_thresholds.csv')
df2 = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\tryout_LSTM_thr2.csv')
y_test_df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\y_test.csv')

# get model from file
save_path = './LSTMmodel.h5'
model = keras.models.load_model(save_path)

with open("X_test.bin", "rb") as data:
    X_test = pickle.load(data)

# predict
print("Predicting...")
predicted = model.predict(X_test)
print("Reshaping predicted")
predicted = np.reshape(predicted, (predicted.size,))
predicted = predicted[1:]

# make lists for calculations result
anomaly_list = []
threshold_list = []
y_test = []

for ij in range(20007, 24989):
    anomaly_list.append(threshold_df.Anomaly[ij])

for ij in range(19993, 24975):
    threshold_list.append(df2.New_threshold[ij])

for ij in range(len(y_test_df)):
    y_test.append(y_test_df.y_test[ij])

# calculate TP, TN, FN, and FP
True_positives = 0
True_negatives = 0
False_negatives = 0
False_positives = 0


LS_anomaly_predictions = []
LS_anomaly_values = []
LS_lower_thresh = (np.array(threshold_list)*0.6)

for orginal, anomaly, predict, thresh in zip(y_test, anomaly_list, predicted, LS_lower_thresh):
    if anomaly and predict >= thresh:
        True_positives = True_positives + 1
        LS_anomaly_predictions.append(predict)
        LS_anomaly_values.append(orginal)
    elif not anomaly and predict < thresh:
        True_negatives = True_negatives + 1
    elif anomaly and predict < thresh:
        False_negatives = False_negatives + 1
        LS_anomaly_predictions.append(predict)
        LS_anomaly_values.append(orginal)
    else:
        False_positives  = False_positives + 1

print("True_negatives: ", True_negatives)
print("True_positives: ", True_positives)
print("False_negatives: ", False_negatives)
print("False_positives: ", False_positives)

try:
    # Calculate precision, recall, and the F1-score
    print("incorrect: ", False_negatives + False_positives)
    precision = True_positives/(True_positives + False_positives)
    print("precision: ", precision)
    recall = True_positives/(True_positives + False_negatives)
    print("recall: ", recall)
    F1_score = 2*True_positives/(2*True_positives + False_positives + False_negatives)
    print("F1_score: ", F1_score)
except:
    print("error in calculations")
try:
    plt.title("Actual Test Values VS Predicted Values")
    
    LSTM_predicted = predicted
    # plt.plot(X_train_list, 'p', label='X_train')
except Exception as e:
    print("plotting exception")
    print(str(e))
print("Total time : ", time.time() - begin)


# Simple RNN

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
import keras

df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\Dataset\jips_dmch_uur_nogaps.csv')
threshold_df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\new_labels_with_thresholds.csv')
df2 = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\tryout_Simple_RNN_thr2.csv')
y_test_df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\y_test.csv')

# get model from file
save_path = './Simple_RNNmodel.h5'
model = keras.models.load_model(save_path)

with open("X_test.bin", "rb") as data:
    X_test = pickle.load(data)

# predict
print("Predicting...")
predicted = model.predict(X_test)
print("Reshaping predicted")
predicted = np.reshape(predicted, (predicted.size,))
predicted = predicted[1:]

# make lists for calculations result
anomaly_list = []
threshold_list = []
y_test = []

for ij in range(20007, 24989):
    anomaly_list.append(threshold_df.Anomaly[ij])

for ij in range(19993, 24975):
    threshold_list.append(df2.New_threshold[ij])

for ij in range(len(y_test_df)):
    y_test.append(y_test_df.y_test[ij])

# calculate TP, TN, FN, and FP
True_positives = 0
True_negatives = 0
False_negatives = 0
False_positives = 0


SR_anomaly_predictions = []
SR_anomaly_values = []
SR_lower_thresh = (np.array(threshold_list)*0.7)

for orginal, anomaly, predict, thresh in zip(y_test, anomaly_list, predicted, SR_lower_thresh):
    if anomaly and predict >= thresh:
        True_positives = True_positives + 1
        SR_anomaly_predictions.append(predict)
        SR_anomaly_values.append(orginal)
    elif not anomaly and predict < thresh:
        True_negatives = True_negatives + 1
    elif anomaly and predict < thresh:
        False_negatives = False_negatives + 1
        SR_anomaly_predictions.append(predict)
        SR_anomaly_values.append(orginal)
    else:
        False_positives  = False_positives + 1

print("True_negatives: ", True_negatives)
print("True_positives: ", True_positives)
print("False_negatives: ", False_negatives)
print("False_positives: ", False_positives)
try:
    # Calculate precision, recall, and the F1-score
    print("incorrect: ", False_negatives + False_positives)
    precision = True_positives/(True_positives + False_positives)
    print("precision: ", precision)
    recall = True_positives/(True_positives + False_negatives)
    print("recall: ", recall)
    F1_score = 2*True_positives/(2*True_positives + False_positives + False_negatives)
    print("F1_score: ", F1_score)
except:
    print("error in calculations")
try:
    # plot all results together
    plt.title("Original Values VS Predicted Values")
    plt.plot(y_test[:len(y_test)], 'y', label='Original values')
    plt.plot(Transformer_predicted[:len(y_test)], 'g', label='Transformer')
    plt.plot(LSTM_predicted[:len(y_test)], 'b', label='LSTM')
    plt.plot(predicted[:len(y_test)], 'darkorange', label='Simple RNN')
    # plt.plot(threshold_list, 'm', label='Threshold')
    # plt.plot(lower_thresh, 'm', label='Lower Threshold')
    # plt.plot(anomaly_list, 'y', label='Anomalies (0 if normal data point, 1 if anomaly)')
    plt.xlabel("Data points")
    plt.ylabel("Transformed Number of Error Messages")
    plt.legend()
    plt.show()
except Exception as e:
    print("plotting exception")
    print(str(e))

# calculate difference between original data point and prediction when data point is anomaly and add them toghether
def subtract(L_pred, L_val):
    difference = 0
    complete_val = 0
    for x1, x2 in zip(L_pred, L_val):
        difference = difference + abs((x2 - x1))
        complete_val = complete_val + x2
    return difference, complete_val

Tr_res, complete_val = subtract(Tr_anomaly_predictions, Tr_anomaly_values)
print(Tr_res)
LS_res, complete_val = subtract(LS_anomaly_predictions, LS_anomaly_values)
print(LS_res)
SR_res, complete_val = subtract(SR_anomaly_predictions, SR_anomaly_values)
print(SR_res)
print(complete_val)