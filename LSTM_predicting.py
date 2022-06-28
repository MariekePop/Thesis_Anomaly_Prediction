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

save_path = './LSTMmodel.h5'
model = keras.models.load_model(save_path)

with open("X_test.bin", "rb") as data:
    X_test = pickle.load(data)
print("Predicting...")
predicted = model.predict(X_test)
print("Reshaping predicted")
predicted = np.reshape(predicted, (predicted.size,))
predicted = predicted[1:]

anomaly_list = []
threshold_list = []
y_test = []
# 1007, 1489
# 15007, 24989
# 20007, 24989
for ij in range(20007, 24989):
    anomaly_list.append(threshold_df.Anomaly[ij])

# 993, 1475
# 14993, 24975
# 19993, 24975
for ij in range(19993, 24975):
    threshold_list.append(df2.New_threshold[ij])

for ij in range(len(y_test_df)):
    y_test.append(y_test_df.y_test[ij])


True_positives = 0
True_negatives = 0
False_negatives = 0
False_positives = 0



lower_thresh = (np.array(threshold_list)*0.6)

for anomaly, predict, thresh in zip(anomaly_list, predicted, lower_thresh):
    if anomaly and predict >= thresh:
        True_positives = True_positives + 1
    elif not anomaly and predict < thresh:
        True_negatives = True_negatives + 1
    elif anomaly and predict < thresh:
        False_negatives = False_negatives + 1
    else:
        False_positives  = False_positives + 1

print("True_negatives: ", True_negatives)
print("True_positives: ", True_positives)
print("False_negatives: ", False_negatives)
print("False_positives: ", False_positives)

try:
    print("incorrect: ", False_negatives + False_positives)
    accuracy = (True_negatives+True_positives)/(True_positives + True_negatives + False_negatives + False_positives)
    print("accuracy: ", accuracy)
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
    
    plt.plot(y_test[:len(y_test)], 'b', label='Original values')
    plt.plot(predicted[:len(y_test)], 'g', label='Predicted values')
    plt.plot(threshold_list, 'r', label='Threshold')
    plt.plot(lower_thresh, 'm', label='Lower Threshold')
    plt.plot(anomaly_list, 'y', label='Anomalies (0 if normal data point, 1 if anomaly)')
    plt.legend()
    plt.show()
    # plt.plot(X_train_list, 'p', label='X_train')
except Exception as e:
    print("plotting exception")
    print(str(e))
print("Total time : ", time.time() - begin)