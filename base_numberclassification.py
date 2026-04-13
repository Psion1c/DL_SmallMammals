import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. DATA PREPARATION STAGE 
# ==========================================

# Load training and testing images
(fine_train, fine_Trlabels), (fine_test, fine_TsLabels) = keras.datasets.cifar100.load_data(label_mode='fine')
(coarse_train, coarse_TrLabels), (coarse_test, coarse_TsLabels) = keras.datasets.cifar100.load_data(label_mode='coarse')

print('Coarse Class: {}'.format(np.unique(coarse_TrLabels)))
print('Fine Class for all: {}'.format(np.unique(fine_Trlabels)))

# Extract all images of a specific coarse class from the TRAINING DATASET
TARGET_COARSE_CLASS = 16 # 16 is "Small Mammals"

idx = []
for i in range(len(coarse_TrLabels)):
    if coarse_TrLabels[i] == TARGET_COARSE_CLASS:
        idx.append(i)

print('Total images with 16 coarse label (Small Mammals) from TRAINING DATASET: {}'.format(len(idx)))

# Extract all image and corresponding "fine" label
train_images, train_labels = fine_train[idx], fine_Trlabels[idx]
print("Shape of the image training dataset: {}".format(train_images.shape))

uniq_fineClass = np.unique(train_labels)
print('Fine Class for the extracted training images: {}'.format(uniq_fineClass))

# Extract all images of a specific coarse class from the TESTING DATASET
idx_test = []
for i in range(len(coarse_TsLabels)):
    if coarse_TsLabels[i] == TARGET_COARSE_CLASS:
        idx_test.append(i)

print('Total images with 16 coarse label (Small Mammals) from TESTING DATASET: {}'.format(len(idx_test)))

# Extract all image and corresponding "fine" label for testing
test_images, test_labels = fine_test[idx_test], fine_TsLabels[idx_test]
print("Shape of the image testing dataset: {}".format(test_images.shape))

# Relabel training and testing dataset to start from zero (0).
for i in range(len(uniq_fineClass)):
    for j in range(len(train_labels)):
        if train_labels[j] == uniq_fineClass[i]:
            train_labels[j] = i
    for j in range(len(test_labels)):
        if test_labels[j] == uniq_fineClass[i]:
            test_labels[j] = i

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# ==========================================
# 2. BASE MODEL BUILD & TRAINING
# ==========================================

model = tf.keras.Sequential()
# 32 convolution filters used each of size 3x3
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)))
# choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2,2)))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
# output a softmax to squash the matrix into output probabilities
model.add(Dense(len(uniq_fineClass), activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train for only 10 epochs as per the base instructions
metricInfo = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# ==========================================
# 3. EVALUATION
# ==========================================

print("\nEvaluating model on unseen test data...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

print("=========================================")
print(f" FINAL OVERALL ACCURACY: {test_acc * 100:.2f}% ")
print("=========================================\n")

# ==========================================
# 4. VISUALIZATIONS
# ==========================================
str_class = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']

# A. LOSS GRAPH
loss = metricInfo.history['loss']
val_loss = metricInfo.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(8, 6), dpi=100)
plt.plot(epochs, loss, 'g-', label="Training loss")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
# Added accuracy to the title of the graph
plt.title(f'Training vs Validation loss (Base Model)\nOverall Test Accuracy: {test_acc * 100:.2f}%', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# B. MULTI-SAMPLE GRID (10 Images)
predictions = model.predict(test_images)
y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = test_labels.flatten()

plt.figure(figsize=(15, 8), dpi=120)
# Added accuracy to the title of the image grid
plt.suptitle(f"Base Model - 10 Test Image Samples\nOverall Accuracy: {test_acc * 100:.2f}%", fontsize=16, fontweight='bold')

for i in range(10): 
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i], interpolation='spline36')
    pred_idx = y_pred_classes[i]
    confidence = predictions[i][pred_idx] * 100
    color = 'green' if pred_idx == y_true_classes[i] else 'red'
    
    # Text layout for P (Prediction) and A (Actual)
    plt.title(f"P: {str_class[pred_idx]} ({confidence:.1f}%)\nA: {str_class[y_true_classes[i]]}", 
              color=color, fontsize=10, fontweight='bold')
    plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.show()