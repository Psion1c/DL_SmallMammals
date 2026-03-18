import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

# DATA PREPARATION STAGE
(coarse_train, coarse_TrLabels), (coarse_test, coarse_TsLabels) = keras.datasets.cifar100.load_data(label_mode='coarse')
(fine_train, fine_Trlabels), (fine_test, fine_TsLabels) = keras.datasets.cifar100.load_data(label_mode='fine')

# Extract specific coarse class (Small Mammals = 16)
TARGET_COARSE_CLASS = 16
idx_train = [i for i in range(len(coarse_TrLabels)) if coarse_TrLabels[i] == TARGET_COARSE_CLASS]
print('Total small mammal images from TRAINING DATASET: {}'.format(len(idx_train)))

train_images, train_labels = fine_train[idx_train], fine_Trlabels[idx_train]
uniq_fineClass = np.unique(train_labels)

idx_test = [i for i in range(len(coarse_TsLabels)) if coarse_TsLabels[i] == TARGET_COARSE_CLASS]
test_images, test_labels = fine_test[idx_test], fine_TsLabels[idx_test]

# Relabel training and testing dataset to start from zero (0)
for i in range(len(uniq_fineClass)):
    for j in range(len(train_labels)):
        if train_labels[j] == uniq_fineClass[i]: train_labels[j] = i
    for j in range(len(test_labels)):
        if test_labels[j] == uniq_fineClass[i]: test_labels[j] = i

# Build the model
model = tf.keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(uniq_fineClass), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train the model
metricInfo = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# Plot training vs validation loss
loss = metricInfo.history['loss']
val_loss = metricInfo.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.clf()
plt.plot(epochs, loss, 'g-', label="Training loss")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training vs Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show() # Commented out so it doesn't pause the script before presentation visuals

# Test the model
str_class = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

classification = model.predict(test_images)

# ==========================================
# PRESENTATION VISUALS
# ==========================================
IMAGE_INDEX = 0
test_img = test_images[IMAGE_INDEX]
true_label_idx = test_labels[IMAGE_INDEX][0]
predictions_array = classification[IMAGE_INDEX]

# 1. VISUALIZE CONFIDENCE PROBABILITIES
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(test_img, cmap=plt.cm.binary)
plt.title(f"Actual Image: {str_class[true_label_idx]}")
plt.axis('off')

plt.subplot(1, 2, 2)
bars = plt.bar(str_class, predictions_array, color='gray')
plt.title("Model's Prediction Confidence")
plt.ylabel("Probability")
plt.ylim([0, 1])

predicted_label_idx = np.argmax(predictions_array)
bars[predicted_label_idx].set_color('red')
bars[true_label_idx].set_color('green')
plt.tight_layout()
plt.show()

# 2. VISUALIZE CONVOLUTIONAL FEATURE MAPS
layer_outputs = [layer.output for layer in model.layers[:1]]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
img_tensor = np.expand_dims(test_img, axis=0)
activations = activation_model.predict(img_tensor)

plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(activations[0, :, :, i], cmap='viridis')
    plt.axis('off')
    plt.title(f'Filter {i+1} Output')
plt.suptitle("What the First Hidden Layer 'Sees'", fontsize=16)
plt.tight_layout()
plt.show()