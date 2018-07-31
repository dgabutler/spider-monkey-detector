#!/usr/bin/env python3

## VGG16 ConvNet Transfer Learning

# Imports
import numpy as np
import os
import subprocess
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping


# Let's create our training and test set.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    '../Data/image_sorted/training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    '../Data/image_sorted/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Now let's download and create a pretrained model.
base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(224, 224, 3))

# Now let's complete the model by adding the fully connected layers.

# Make add the fully connected layers
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
custom_out = Dense(21, activation='softmax')(x)

# Create the final architecture.
model = Model(inputs=base_model.input, outputs=custom_out)
print(model.summary())

# I'll be using a dynamic learning rate (stepwise annealing).
def scheduler(epoch):
    if epoch <= 5:
        lr = 0.00001
        return lr
    elif epoch > 5 and epoch <= 10:
        lr = 0.00001
        return lr
    elif epoch > 10 and epoch <= 20:
        lr = 0.000005
        return lr
    elif epoch > 20 and epoch <= 30:
        lr = 0.000001
        return lr
    elif epoch > 30 and epoch <= 35:
        lr = 0.0000005
        return lr
    elif epoch > 35 and epoch <= 40:
        lr = 0.0000001
        return lr
    else:
        lr = 0.00001
        return lr

dynamic_lr = LearningRateScheduler(scheduler, verbose=1)


#  Save the model and weights whenever there is an improvement in accuracy
filepath = "../Data/vgg16_checkpoint.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# Saving the LearningRateScheduler and the Checkpoint as list for callback
callbacks_list = [checkpoint, dynamic_lr]

# Compiling the model
model.compile(optimizer="Adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# fitting the model
model.fit_generator(training_set,
                    # steps_per_epoch=422,
                    epochs=40,
                    validation_data=test_set,
                    callbacks=callbacks_list)
