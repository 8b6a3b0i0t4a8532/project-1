# project-1
import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50V2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURE YOUR LOCAL PATH HERE
# ---------------------------------------------------------
# If your 'archive' folder is in the same place as this script:
base_dir = r'archive\PLD_3_Classes_256' 

# If you need to use an absolute path (e.g., C:/Users/...), uncomment and edit below:
# base_dir = r'C:\Users\YourName\Documents\Project\archive\PLD_3_Classes_256'
# ---------------------------------------------------------

train_dir = os.path.join(base_dir, 'Training')
val_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Testing')

print(f"Dataset Base Directory: {base_dir}")
print(f"Training Directory: {train_dir}")

# Verify one folder exists to ensure path is correct
if os.path.exists(train_dir):
    print("✅ Path found successfully!")
else:
    print("❌ Path not found. Please check 'base_dir' variable.")

# ==========================================
# CELL 2: VISUALIZATION OF DATASET
# ==========================================

Early_Blight_path = os.path.join(train_dir, 'Early_Blight')
Healthy_path = os.path.join(train_dir, 'Healthy')
Late_Blight_path = os.path.join(train_dir, 'Late_Blight')

# Get list of images
image_files_early = [os.path.join(Early_Blight_path, f) for f in os.listdir(Early_Blight_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files_healthy = [os.path.join(Healthy_path, f) for f in os.listdir(Healthy_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files_late = [os.path.join(Late_Blight_path, f) for f in os.listdir(Late_Blight_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

num_samples = 3
selected_early = random.sample(image_files_early, num_samples)
selected_healthy = random.sample(image_files_healthy, num_samples)
selected_late = random.sample(image_files_late, num_samples)

# Create subplots
fig, axes = plt.subplots(3, num_samples, figsize=(15, 10))
fig.suptitle('Potato Leaf Disease Samples', fontsize=24)

def plot_images(images, row, title):
    for i, img_path in enumerate(images):
        ax = axes[row, i]
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=15)
        ax.axis('off')

plot_images(selected_early, 0, 'Early Blight')
plot_images(selected_healthy, 1, 'Healthy')
plot_images(selected_late, 2, 'Late Blight')

plt.tight_layout()
plt.show()

# ==========================================
# CELL 3: DATA GENERATORS (PREPROCESSING)
# ==========================================

BATCH_SIZE = 32
IMG_SIZE = (224, 224) # Resized for VGG/ResNet

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("\n--- Generators ---")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)

# ==========================================
# CELL 4: BUILD MODEL 1 (VGG16 Transfer)
# ==========================================

def build_vgg_model():
    print("Building VGG16 Model...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model_vgg = build_vgg_model()

# ==========================================
# CELL 5: TRAIN MODEL 1
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

print("Training VGG16...")
# Note: Reduced epochs slightly for local machine testing
history_vgg = model_vgg.fit(
    train_generator,
    epochs=10, 
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# ==========================================
# CELL 6: BUILD MODEL 2 (ResNet50V2 Transfer)
# ==========================================

def build_resnet_model():
    print("Building ResNet50V2 Model...")
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model_resnet = build_resnet_model()

# ==========================================
# CELL 7: TRAIN MODEL 2
# ==========================================

print("Training ResNet50V2...")
history_resnet = model_resnet.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# ==========================================
# CELL 8: ENSEMBLE PREDICTION
# ==========================================

print("Generating Ensemble Predictions...")
test_generator.reset()
pred_vgg = model_vgg.predict(test_generator, verbose=1)

test_generator.reset()
pred_resnet = model_resnet.predict(test_generator, verbose=1)

# Average predictions
final_preds = (0.5 * pred_vgg) + (0.5 * pred_resnet)
final_class_indices = np.argmax(final_preds, axis=1)
true_classes = test_generator.classes

acc = np.sum(final_class_indices == true_classes) / len(true_classes)
print(f"Ensemble Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(true_classes, final_class_indices)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Ensemble Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Save locally
model_vgg.save('potato_vgg16.h5')
model_resnet.save('potato_resnet50.h5')
print("Models saved to local directory.")

# ==========================================
# CELL 9: TEST SINGLE IMAGE
# ==========================================

# Pick a random image from the test set folder to predict
test_folder_early = os.path.join(test_dir, 'Early_Blight')
if os.path.exists(test_folder_early):
    random_img_name = random.choice(os.listdir(test_folder_early))
    img_path_test = os.path.join(test_folder_early, random_img_name)
    
    img = keras.utils.load_img(img_path_test, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    p1 = model_vgg.predict(img_array)
    p2 = model_resnet.predict(img_array)
    avg_pred = (p1 + p2) / 2.0
    
    print(f"Predicted Class: {class_names[np.argmax(avg_pred)]}")
    print(f"Confidence: {np.max(avg_pred)*100:.2f}%")
