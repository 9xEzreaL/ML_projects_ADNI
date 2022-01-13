"""
all 224 is original 176
"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from distutils.dir_util import copy_tree, remove_tree
from PIL import Image
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.client import device_lib


num_class= 3 # 2 , 3
backbone = InceptionV3  # DenseNet121 VGG16 InceptionV3
lr = 0.05
EPOCHS = 7
BRIGHT_RANGE = [0.1, 0.5]
# tf.keras.optimizers.SGD(lr ,momentum=0.5)
# tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-07)
# tf.keras.optimizers.Adagrad(learning_rate=lr, epsilon=1e-07)
# # tf.keras.optimizers.Adagrad(learning_rate=lr, epsilon=1e-07)
optim = tf.keras.optimizers.SGD(lr=lr ,momentum=0.5)

# base_dir = "Alzheimer_s Dataset/"
base_dir = 'alzdataset_{}class/'.format(num_class)
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset_{}class/".format(num_class)
backbone_name = backbone
if os.path.exists(work_dir):
    remove_tree(work_dir)

os.mkdir(work_dir)
# train, test combine together
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)

print("Working Directory Contents:", os.listdir(work_dir))
# WORK_DIR = './dataset/'
WORK_DIR = './dataset_{}class/'.format(num_class)
test_WORK_DIR = './ADNI_{}class'.format(num_class)

# CLASSES = [ 'NonDemented',
#             'VeryMildDemented',
#             'MildDemented',
#             'ModerateDemented']
if num_class ==2:
    CLASSES = [ 'NonDemented',
                'ModerateDemented']
    test_CLASSES = ['CN',
                    'AD']

elif num_class ==3:
    CLASSES = ['NonDemented',
               'MildDemented',
               'ModerateDemented']
    test_CLASSES = ['CN',
                    'MCI',
                    'AD']

IMG_SIZE = 224
IMAGE_SIZE = [224, 224]
DIM = (IMG_SIZE, IMG_SIZE)

# Performing Image Augmentation to have more data samples

ZOOM = [.99, 1.01]
# BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
VERT_FLIP =True

FILL_MODE = "constant"
DATA_FORMAT = "channels_last"
"""
horizontal filp: random flip
brightness range: random make img dark or bright
zoom range: some pixel height,width random large or small
"""

work_dr = IDG(rescale = 1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP, vertical_flip=VERT_FLIP)
test_work_dr = IDG(rescale= 1./255)
train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)
test_data_gen = test_work_dr.flow_from_directory(directory=test_WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)
def show_images(generator, y_pred=None):
    """
    Input: An image generator,predicted labels (optional)
    Output: Displays a grid of 9 images with lables
    """
    # get image lables
    labels = dict(zip([0, 1, 2, 3], CLASSES))

    test_labels = dict(zip([0,1,2],test_CLASSES))
    # get a batch of images
    x, y = generator.next()

    # display a grid of 9 images
    plt.figure(figsize=(10, 10))
    if y_pred is None:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            idx = randint(0, 3264)
            plt.imshow(x[idx])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[idx])]))
            plt.savefig('Fig/{}_brain(3pic)_bri{}.png'.format(backbone_name, BRIGHT_RANGE))

    else:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Actual:{} \nPredicted:{}".format(labels[np.argmax(y[i])], labels[y_pred[i]]))
            plt.savefig('Fig/{}_brain(3pic)_bri{}.png'.format(backbone_name, BRIGHT_RANGE))
# Display Train Images
show_images(train_data_gen)

train_data, train_labels = train_data_gen.next()
test_data, test_labels = test_data_gen.next()
print(train_data.shape, train_labels.shape, type(train_labels))
print(test_data.shape, test_labels.shape, type(test_labels))
# print("test_label",test_labels)
# make every stage the same
sm = SMOTE(random_state=42)

# print('train_labels: ',train_labels)
train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)

if num_class ==2:
# for 2 class use
    Train_labels=[]
    for i in range(len(train_labels)):
        if train_labels[i]==[0]:
            Train_labels.append([1.,0.])
        else:
            Train_labels.append([0.,1.])
    train_labels = np.array(Train_labels)


train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print('train_labels: ',len(train_labels))
print(train_data.shape, train_labels.shape)
# train_data, test_data, train_labels, test_labels = train_test_split(test_data, test_labels, test_size = 1, random_state=42)
# train_data type : np.array
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)

if backbone == VGG16:
    model = VGG16(input_shape=(224,224,3), include_top=False, weights="imagenet")
    for layer in model.layers:
        layer.trainable=False

elif backbone == DenseNet121:
    model = DenseNet121(input_shape=(224,224,3), include_top=False, weights="imagenet")
    for layer in model.layers:
        layer.trainable=False

elif backbone ==InceptionV3:
    model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    for layer in model.layers:
        layer.trainable=True

custom_inception_model = Sequential([
        model,
        Dropout(0.5),
        GlobalAveragePooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
    # class2
        Dense(num_class, activation='softmax')
    ], name = "inception_cnn_model")


# Defining a custom callback function to stop training our model when accuracy goes above 99%

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True


my_callback = MyCallback()
# ReduceLROnPlateau to stabilize the training process of the model
rop_callback = ReduceLROnPlateau(monitor="val_loss", patience=3)

# original num_class=4
METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'),
           tfa.metrics.F1Score(num_classes=num_class)]

CALLBACKS = [my_callback, rop_callback]

custom_inception_model.compile(optimizer=optim,
                               loss=tf.losses.CategoricalCrossentropy(),
                               metrics=METRICS)

custom_inception_model.summary()

#Fit the training data to the model and validate it using the validation data
print(train_data.shape, train_labels.shape)
print(val_data.shape, val_labels.shape)
print(test_data.shape, test_labels.shape)
history = custom_inception_model.fit(train_data, train_labels, batch_size=32, validation_data=(val_data, val_labels), callbacks=CALLBACKS, epochs=EPOCHS)


test_scores = custom_inception_model.evaluate(test_data, test_labels)
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))
pred_labels = custom_inception_model.predict(test_data)

def roundoff(arr):
    """To round off according to the argmax of each predicted label array. """
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))

#Plot the confusion matrix to understand the classification in detail

pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)

conf_arr = confusion_matrix(test_ls, pred_ls)

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels= CLASSES,
                yticklabels=CLASSES)
plt.title('Alzheimer\'s Disease Diagnosis')
plt.xlabel('Prediction')
plt.ylabel('Truth')
# plt.show()
plt.savefig('Fig/{}_brain(3pic)_matrix{}.png'.format(backbone_name,BRIGHT_RANGE))

#Printing some other classification metrics
print("Balanced Accuracy Score: {} %".format(round(BAS(test_ls, pred_ls) * 100, 2)))
print("Matthew's Correlation Coefficient: {} %".format(round(MCC(test_ls, pred_ls) * 100, 2)))
#Saving the model for future use
custom_inception_model_dir = work_dir + "alzheimer_inception_cnn_model"
custom_inception_model.save(custom_inception_model_dir, save_format='h5')
os.listdir(work_dir)
pretrained_model = tf.keras.models.load_model(custom_inception_model_dir)

