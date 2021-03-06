""" Inspired by example from
https://keras.io/examples/timeseries/timeseries_classification_transformer/
Uses the Keras (TensorFlow) backend
The basic idea is to predict anomalies in a time-series.
"""
import time
begin = time.time()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
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

# TransformerEncoder block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=4)(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=3)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# transformer model
def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
       x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation = "linear")(x)
 
    model = keras.Model(inputs, outputs)
    start = time.time()
    print("Compilation Time : ", time.time() - start)
    print("model: ", model)
    model.summary()
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()
    threshold_input = []
    threshold_result = []
    date_time_data_df = []

    if data is None:
        print('Loading data... ')
        # split data into training and testset
        # X_train, y_train, X_test, y_test, threshold_result, threshold_input, date_time_data_df = get_split_prep_data(0, 20000, 20001, 25000, threshold_result, threshold_input, date_time_data_df)
     else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    # Build the model
    if model is None:
        input_shape = X_train.shape[1:]
        print(input_shape)
        n_classes = len(np.unique(y_train))
        print(n_classes)
        if model is None:
            model = build_model(
                input_shape,
                head_size=256,
                num_heads=4,
                ff_dim=4,
                num_transformer_blocks=2,
                mlp_units=[256],
                mlp_dropout=0,
                dropout=0,
            )
        # compile the model
        model.compile(loss="mse", optimizer="rmsprop")
    try:
        print("Training...")
        # train the model
        model.fit(
                X_train, y_train,
                batch_size=batch_size, callbacks=callbacks, validation_split=0.1)
        # model.evaluate(X_test, y_test, verbose=1)
    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    # save the model
    save_path = './Transformermodel.h5'
    model.save(save_path)

    # store values in files
    df2 = pd.DataFrame({
        'New_threshold': threshold_result,
    })
    df2.to_csv('tryout_Transformer_thr2.csv')

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