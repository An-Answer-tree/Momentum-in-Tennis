import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime


# Read Dataset to Dataframe
df = pd.read_csv('./question3/dataset.csv', header=None)
print(df.head())

# Divide Features and Labels
X = df.iloc[:, 0:7]
Y = df.iloc[:, 7]


# Transform Dataframe to Numpy Array
X = X.values
Y = Y.values
X, Y = shuffle(X, Y, random_state=42)


# divide train_X, train_Y, test_X, test_Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Features Data Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Create BPNN Model Using Keras
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')  # 二分类问题，输出层使用sigmoid激活函数
# ])


# # Compile Model
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',
#                                                                         tf.keras.metrics.Precision(),
#                                                                         tf.keras.metrics.Recall(),
#                                                                         tf.keras.metrics.AUC()
#                                                                         ])


# # 设置 TensorBoard 回调
# log_dir = "./logs"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# # Train Model
# model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), callbacks=[tensorboard_callback])
# model.save('./question3/saved_model')


# Load Model
model = tf.keras.models.load_model(
    './question3/saved_model',
    custom_objects=None, compile=True)


# Predict Data in a Match
data_predict = pd.read_csv('./question3/5_1304.csv', header=None)

# Get Features
X_predict = data_predict.iloc[:, 0:7]

# Transform dataframe to Numpy Array
X_predict = X_predict.values

# Standardization
X_predict = scaler.transform(X_predict)

# Get Result
predictions = model.predict(X_predict)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Draw
plt.figure(figsize=(18, 8))
plt.axhline(y=0.5, color='orange', linestyle='--')  # 画一条横线
plt.plot(predictions, label='trend', color='black', antialiased=True)
plt.plot(binary_predictions, label='binary_predictions', color='red', antialiased=True)
plt.title('2023-wimbledon-1304 With perturbation Predictions')
plt.xlabel('Total Scores')
plt.ylabel('Probability')
plt.xticks(np.arange(0, 70, step=10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlim(0, 60)
plt.ylim(-0.1, 1.1)
plt.legend()

plt.show()