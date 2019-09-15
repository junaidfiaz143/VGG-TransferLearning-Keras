#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:57:13 2018

@author: masood
"""
import keras
from keras import backend as K
from keras.utils import np_utils
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
#os.chdir("/home/masood/project")
#img_width, img_height = 299, 299
#path = "dirout20"
os.chdir("F:/Thesis-results-24jun19/all-results/only-pistol-revolver/VGG/VGG/vgg1-scratch")
#img_width, img_height = 224, 224
path = "data2"
num_channel = 3
classes = os.listdir(path)
img_data_list = []

for fol in classes:
    print(fol)
    img_list = os.listdir(path + '//' + fol)
    for img in img_list:
        input_img = cv2.imread(path + '//' + fol + '//' + img)
        input_img = cv2.resize(input_img,(299,299))
        img_data_list.append(input_img)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)
if num_channel == 1:
	if K.image_dim_ordering() == 'th':
		img_data = np.expand_dims(img_data, axis=1) 
		print(img_data.shape)
	else:
		img_data = np.expand_dims(img_data, axis=4) 
		print(img_data.shape)
		
else:
	if K.image_dim_ordering() == 'th':
		img_data = np.rollaxis(img_data,3,1)
print(img_data.shape)


nb_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype = 'int64')
labels[0:981] = 0
labels[982:1731]=1
#labels[982:1731] = 1
"""labels[0:307] = 0
labels[308:509] = 1
labels[510:890] = 2
labels[891:1119] = 3
labels[1120:1601] = 4
labels[1602:1858] = 5
labels[1859:2127] = 6
labels[2128:2279] = 7
labels[2280:2659] = 8
labels[2660:3035] = 9
labels[3036:3338] = 10
labels[3339:3624] = 11
labels[3625:3853] = 12
labels[3854:4295] = 13
labels[4296:4609] = 14
labels[4610:4888] = 15
labels[4889:5252] = 16
labels[5253:5692] = 17
labels[5693:5974] = 18
labels[5975:6367] = 19
labels[6368:6759] = 20
names = ["n000002","n000003","n000004","n000005","n000006","n000007","n000008","n000010","n000011","n000012","n000013","n000014","n000015","n000016","n000017","n000018","n000019","n000020","n000021",
           "n000022"]
#names=["pistol","non-pistol"]"""
names=["not-pistol","pistol"]
#names=["pistol"]
Y = np_utils.to_categorical(labels, nb_classes)
#x,y = shuffle(img_data,Y, random_state = 2)
#X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2)

x,y = shuffle (img_data,Y, random_state = 2)
X_train, X_test, y_train, y_test = train_test_split (x,y, test_size = 0.15, random_state = 2)
x=None
y= None
X_train, X_val, y_train, y_val = train_test_split (X_train,y_train, test_size = 0.15, random_state = 42)
input_shape = img_data[0].shape
z = keras.layers.Input(input_shape)
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=z)
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=z)
#base_model=InceptionResNetV2(weights='imagenet', include_top=False,
#input_tensor=z)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and
                                     #classify for better results.
x = Dense(1024,activation='relu')(x) #dense layer 2
Dropout(0.25)
x = Dense(512,activation='relu')(x) #dense layer 3
x = Dense(1024,activation='relu')(x)
Dropout(0.25)
predictions = Dense(2,activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False





# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics = ['accuracy'])
#model.compile(optimizer='sgd', loss='binary_crossentropy', metrics = ['accuracy'])
filepath = "dfv.h5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]
batch_size = 16
epochs = 16
history = model.fit(X_train,y_train, batch_size=batch_size,epochs=epochs,verbose=1, callbacks=callbacks_list,validation_data=(X_val, y_val))
scores = model.evaluate(X_test, y_test, verbose=1)
print("\n\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))
print("Baseline Error: %.2f%%\n" % (100 - scores[1] * 100))
print("Test Accuracy is:%.2f%%" % (scores[1] * 100))
#model_yaml = model.to_yaml()
#with open("dfv.yaml", "w") as yaml_file:
#    yaml_file.write(model_yaml)
print("Saved model to disk")
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print('break here')
plt.clf()
print('break here')
# summarize history for loss
plt2.plot(history.history['loss'])
plt2.plot(history.history['val_loss'])
plt2.title('model loss')
plt2.ylabel('loss')
plt2.xlabel('epoch')
plt2.legend(['train', 'val'], loc='upper left')
plt2.show()
pred_Y = model.predict(X_test)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    y_pred = (y_pred > 0.5) 
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    # Only use the labels that appear in the data
    # classes=["pistol","non-pistol"]
    #classes = names = ["n000002","n000003","n000004","n000005","n000006","n000007","n000008","n000010","n000011","n000012","n000013","n000014","n000015","n000016","n000017","n000018","n000019","n000020","n000021",
     #      "n000022"]
    classes=["not-pistol","pistol"]
    #classes=["pistol"]
    #classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ...  and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, pred_Y, classes=names,
                      title='Confusion matrix')
print('Complete')
