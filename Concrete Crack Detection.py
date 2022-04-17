# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:16:58 2022

@author: syafi
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib

file_path = r"C:\Users\syafi\Downloads\Concrete Crack Images for Classification\Concrete Crack Images for Classification"
PATH = pathlib.Path(file_path)

#%%

BATCH_SIZE = 512
IMG_SIZE = (160,160)
dataset = tf.keras.utils.image_dataset_from_directory(PATH,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)

#%%
train_batches = tf.data.experimental.cardinality(dataset)
val_dataset = dataset.take(train_batches//5)
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)
train_dataset = dataset.skip(train_batches//5)

#%%

class_names = dataset.class_names
plt.figure(figsize=(10,10))
for images, labels in dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8")) 
        plt.title(class_names[labels[i]])
        plt.axis("off")

#%%

#To create prefetch dataset for better performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%

data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
data_augmentation.add(tf.keras.layers.RandomRotation(0.2))

#%%

#Show example of applied image augmentation
for images,labels in train_dataset_pf.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
        
#%%

#We will make use of the method provided in the pretrained model object to rescale input
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#Create the base model by calling out MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

 #%%

#Freeze the base model
base_model.trainable = False
base_model.summary()

#%%
#Add classification layer using global average pooling
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

#Add output layer
prediction_layer = tf.keras.layers.Dense(1)
#%%

#Use functional API to create the entire model (input pipeline + NN)
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs,outputs)
model.summary()

#%%

#Compile model
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=adam,loss=loss,metrics=['accuracy'])

#%%

#Model performance before training
loss0,accuracy0 = model.evaluate(validation_dataset_pf)
print("---------------------Before Training----------------------")
print(f"Loss = {loss0}")
print(f"Accuracy = {accuracy0}")

#%%

EPOCHS = 10
import datetime
log_path = r"C:\Users\syafi\Documents\AI Class\Deep Learning\Tensorboard\logs_" + datetime.datetime.now().strftime("%d%m%Y - %H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
history = model.fit(train_dataset_pf,validation_data=validation_dataset_pf,epochs=EPOCHS,callbacks=[tb_callback])

#%%

#Now we perform fine tuning, the goal is to train the higher level convolution layers.
#This is to allow high level features to be adapted to our own dataset

#1. Unfreeze base_model
base_model.trainable = True

#2. Freeze first n number of layers (leave the behind layers to be unfrozen so that they will be trained.)
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.00001)        
model.compile(optimizer=rmsprop,loss=loss,metrics=['accuracy'])
model.summary()
#%%

fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch

#Continue from previous checkpoint
history_fine = model.fit(train_dataset_pf,
                         validation_data=validation_dataset_pf,
                         epochs=total_epoch,initial_epoch=history.epoch[-1],
                         callbacks=[tb_callback])
#%%

#Evaluate with test dataset
test_loss,test_accuracy = model.evaluate(test_dataset_pf)
print('------------------------Test Result----------------------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%

#Deploy model to make prediction
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

#Apply sigmoid to output, since output is in a form of logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5,0,1)

#Compare predictions and labels
print(f'Prediction: {predictions.numpy()}')
print(f'Labels: {label_batch}')

#%%

#Plot the predictions
plt.figure(figsize=(10,10))
for i in range(9):
    axs = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(class_names[predictions[i]])
    plt.axis('off')
    
#%%

import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_mae = history.history['accuracy']
val_mae = history.history['val_accuracy']
epochs = history.epoch

plt.plot(epochs,training_loss,label='Training loss')
plt.plot(epochs,val_loss,label='Validation loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs,training_mae,label='Training accuracy')
plt.plot(epochs,val_mae,label='Validation accuracy')
plt.title('Training accuracy vs Validation accuracy')
plt.legend()
plt.figure()

plt.show()