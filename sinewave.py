import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import numpy as np
import math
import matplotlib.pyplot as plt

# 1. DATA PREPARATION STAGE
SAMPLES = 1000
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
y_values = np.sin(x_values)

TRAIN_SPLIT = int(0.60 * SAMPLES)
TEST_SPLIT = int(0.20 * SAMPLES + TRAIN_SPLIT)

y_noise = y_values.copy()
y_noise += 0.1 * np.random.randn(*y_values.shape)

x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_noise, [TRAIN_SPLIT, TEST_SPLIT])
y_clean = y_values[TEST_SPLIT:]

plt.plot(x_values, y_values, 'b', label='Clean Sine Wave')
plt.plot(x_train, y_train, 'g.', label='Noisy Training Data')
plt.title('Generated Sine Wave Data')
plt.legend()
# plt.show() # Commented out for smooth run

# 2. BUILD THE MODEL
model = tf.keras.Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# 3. TRAIN THE MODEL
metricInfo = model.fit(x_train, y_train, epochs=50, validation_data=(x_validate, y_validate))

loss = metricInfo.history['loss']
val_loss = metricInfo.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show() # Commented out for smooth run

# 4. TEST THE MODEL
predictions = model.predict(x_test)

plt.clf()
plt.plot(x_test, y_clean, 'b', label='Observation')
plt.plot(x_test, predictions, 'r*', label='Predictions')
plt.title('Model Predictions vs Actual Data')
plt.legend()
plt.show()

model.save('baseModel.h5')