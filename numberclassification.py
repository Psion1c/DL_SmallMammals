import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, BatchNormalization, Flatten
import numpy as np
import matplotlib.pyplot as plt

# 1. DATA PREPARATION
(fine_train, fine_Trlabels), (fine_test, fine_TsLabels) = keras.datasets.cifar100.load_data(label_mode='fine')
(coarse_train, coarse_TrLabels), (coarse_test, coarse_TsLabels) = keras.datasets.cifar100.load_data(label_mode='coarse')

TARGET_COARSE_CLASS = 16 
idx_train = [i for i in range(len(coarse_TrLabels)) if coarse_TrLabels[i] == TARGET_COARSE_CLASS]
train_images, train_labels = fine_train[idx_train], fine_Trlabels[idx_train]

idx_test = [i for i in range(len(coarse_TsLabels)) if coarse_TsLabels[i] == TARGET_COARSE_CLASS]
test_images, test_labels = fine_test[idx_test], fine_TsLabels[idx_test]

str_class = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
uniq_fineClass = np.unique(train_labels)
for i in range(len(uniq_fineClass)):
    train_labels[train_labels == uniq_fineClass[i]] = i
    test_labels[test_labels == uniq_fineClass[i]] = i

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 2. STABLE AUGMENTATION
data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
])

# 3. SIMPLIFIED RELIABLE ARCHITECTURE (Back to early success style)
model = models.Sequential([
    Input(shape=(32, 32, 3)),
    data_augmentation,
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Flatten(), # Returning to Flatten for that "Early Version" direct mapping
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(5, activation='softmax')
])

# 4. TRAINING (Fewer epochs for stability)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Using 30 epochs - enough to learn, not enough to overfit
history = model.fit(train_images, train_labels, 
                    epochs=30, 
                    validation_split=0.2, 
                    batch_size=32)

# ==========================================
# 5. VISUALIZATIONS
# ==========================================

# A. LOSS GRAPH (Should show two smooth lines moving down together)
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(history.history['loss'], 'g-', label='Training loss')
plt.plot(history.history['val_loss'], 'b', label='Validation loss')
plt.title('Reliable Training vs Validation Loss', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# B. MULTI-SAMPLE GRID
predictions = model.predict(test_images)
y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = test_labels.flatten()

plt.figure(figsize=(15, 8), dpi=120)
for i in range(10): 
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i], interpolation='spline36')
    pred_idx = y_pred_classes[i]
    confidence = predictions[i][pred_idx] * 100
    color = 'green' if pred_idx == y_true_classes[i] else 'red'
    plt.title(f"P: {str_class[pred_idx]} ({confidence:.1f}%)\nA: {str_class[y_true_classes[i]]}", color=color, fontsize=9, fontweight='bold')
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# C. CONFUSION MATRIX
cm = np.zeros((5, 5), dtype=int)
for i in range(len(y_true_classes)):
    cm[y_true_classes[i], y_pred_classes[i]] += 1

plt.figure(figsize=(8, 7), dpi=100)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Final Confusion Matrix', fontsize=14)
plt.colorbar()
plt.xticks(np.arange(5), str_class, rotation=45)
plt.yticks(np.arange(5), str_class)
for i in range(5):
    for j in range(5):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > (cm.max()/2) else "black")
plt.show()