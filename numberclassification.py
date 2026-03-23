import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, BatchNormalization, Flatten
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = 'mammal_classifier.keras'
HISTORY_PATH = 'training_history.csv'

# ==========================================
# 2. DATA PREPARATION
# ==========================================
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

# ==========================================
# 3. MODEL LOADING / TRAINING
# ==========================================
if os.path.exists(MODEL_PATH) and os.path.exists(HISTORY_PATH):
    print(f"\n--- Loading saved model and history... ---")
    model = keras.models.load_model(MODEL_PATH)
    history_df = pd.read_csv(HISTORY_PATH)
else:
    print("\n--- Starting Smoothed Training... ---")
    
    data_augmentation = keras.Sequential([
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
    ])

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
        
        Flatten(), 
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(5, activation='softmax')
    ])

    # THE FIX: Lowered learning rate from 0.001 to 0.0005 for smoother descent
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(HISTORY_PATH, append=False)

    # THE FIX: Increased batch_size to 64 to stabilize the loss graph
    history = model.fit(train_images, train_labels, 
                        epochs=120, validation_split=0.2, 
                        batch_size=64, 
                        callbacks=[early_stop, lr_scheduler, csv_logger])
    
    model.save(MODEL_PATH)
    history_df = pd.DataFrame(history.history)

# ==========================================
# 4. FINAL EVALUATION & TERMINAL OUTPUT
# ==========================================
print("\nEvaluating model on unseen test data...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

print("=========================================")
print(f" FINAL OVERALL ACCURACY: {test_acc * 100:.2f}% ")
print("=========================================\n")

# ==========================================
# 5. VISUALIZATIONS
# ==========================================

# A. LOSS GRAPH
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(history_df['loss'], 'g-', label='Training loss')
plt.plot(history_df['val_loss'], 'b', label='Validation loss')
plt.title('Training vs Validation Loss', fontsize=14)
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