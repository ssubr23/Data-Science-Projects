import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D,
                                     BatchNormalization, Flatten, Conv2D, AveragePooling2D,
                                     MaxPooling2D, Dropout)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set the training data directory
XRay_Directory = './Chest_X_Ray/train'

# List the folders in the directory
print(os.listdir(XRay_Directory))

# Image generator to generate tensor images data and normalize them
# Using 20% of the data for validation
image_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

# Generate batches of images
# Perform shuffling and image resizing
train_generator = image_generator.flow_from_directory(
    batch_size=40,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode='categorical',
    subset="training"
)

validation_generator = image_generator.flow_from_directory(
    batch_size=40,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode='categorical',
    subset="validation"
)

# Generate a batch of images and labels
train_images, train_labels = next(train_generator)

# Display shapes of images and labels
print(train_images.shape)
print(train_labels.shape)
print(train_labels)

# Labels Translator
label_names = {0: 'Covid-19', 1: 'Normal', 2: 'Viral Pneumonia', 3: 'Bacterial Pneumonia'}

# Create a grid of 36 images along with their corresponding labels
L = 6
W = 6

fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(train_images[i])
    axes[i].set_title(label_names[np.argmax(train_labels[i])])
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)

# Load the ResNet50 model
basemodel = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

# Display the model summary
basemodel.summary()

# Freeze the model up to the last stage-4 and re-train stage-5
for layer in basemodel.layers[:-10]:
    layer.trainable = False

# Add custom layers on top of the base model
headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size=(4, 4))(headmodel)
headmodel = Flatten(name='flatten')(headmodel)
headmodel = Dense(256, activation="relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation="relu")(headmodel)
headmodel = Dropout(0.2)(headmodel)
headmodel = Dense(4, activation='softmax')(headmodel)

# Create the final model
model = Model(inputs=basemodel.input, outputs=headmodel)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),
    metrics=["accuracy"]
)

# Early stopping to exit training if validation loss is not decreasing after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

# Generate training and validation data
train_generator = image_generator.flow_from_directory(
    batch_size=4,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode='categorical',
    subset="training"
)

val_generator = image_generator.flow_from_directory(
    batch_size=4,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode='categorical',
    subset="validation"
)

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // 4,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.n // 4,
    callbacks=[checkpointer, earlystopping]
)

# Plot training accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Loss and Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy and Loss')
plt.legend(['Training Accuracy', 'Training Loss'])
plt.show()

# Plot validation loss
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Cross-Validation')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend(['Validation Loss'])
plt.show()

# Plot validation accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy Progress During Cross-Validation')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend(['Validation Accuracy'])
plt.show()

# Set the test data directory
test_directory = '.Chest_X_Ray/Test'

# Generate test data
test_gen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_gen.flow_from_directory(
    batch_size=40,
    directory=test_directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode='categorical'
)

# Evaluate the model on the test data
evaluate = model.evaluate_generator(test_generator, steps=test_generator.n // 4, verbose=1)
print('Test Accuracy: {}'.format(evaluate[1]))

# Initialize lists for predictions and ground truth
prediction = []
original = []
image = []

# Loop through the test directory and make predictions
for i in range(len(os.listdir(test_directory))):
    for item in os.listdir(os.path.join(test_directory, str(i))):
        img = cv2.imread(os.path.join(test_directory, str(i), item))
        img = cv2.resize(img, (256, 256))
        image.append(img)
        img = img / 255.0
        img = img.reshape(-1, 256, 256, 3)
        predict = model.predict(img)
        predict = np.argmax(predict)
        prediction.append(predict)
        original.append(i)

# Calculate the accuracy score
score = accuracy_score(original, prediction)
print("Test Accuracy: {}".format(score))

# Create a grid of predicted vs actual labels
L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(image[i])
    axes[i].set_title('Guess={}\nTrue={}'.format(str(label_names[prediction[i]]), str(label_names[original[i]])))
    axes[i].axis('off')

plt.subplots_adjust(wspace=1.2)
plt.show()

# Print classification report
print(classification_report(np.asarray(original), np.asarray(prediction)))

# Confusion matrix
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)

# Set labels, title, and display the confusion matrix
ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion Matrix')
plt.show()
