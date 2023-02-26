from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
import skimage
#from keras.layers.advanced_activations import ELU
from tensorflow.keras.layers import ELU
from tensorflow.keras.applications.vgg16 import VGG16


dirname = os.path.join('/content/drive/MyDrive/', 'Oasis90')
imgpath = dirname + os.sep 
 
images = []
directories = []
dircount = []
prevRoot=''
cant=0
 
print("leyendo imagenes de ",imgpath)
 
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1

print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))
 
oasis=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    oasis.append(name[len(name)-1])
    indice=indice+1
 
y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
print(X.shape)

x2 = []

for i in X:
    x2.append(skimage.color.gray2rgb(i))
x2 = np.array(x2)    
print(x2.shape)

X = x2

tf.keras.applications.vgg16.preprocess_input(
    X, data_format=None
)

train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.25)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.
 
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
 
# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])
 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.25)
 
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

INIT_LR = 1e-3
epochs = 100
batch_size = 16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
conv_base.summary()
for layer in conv_base.layers:
    layer.trainable = False
conv_base.summary()

oasis_model = Sequential()
oasis_model.add(conv_base)
oasis_model.add(Flatten())
oasis_model.add(Dense(32, activation='linear'))
oasis_model.add(ELU(alpha=1))
oasis_model.add(Dropout(0.5)) 
oasis_model.add(Dense(3, activation='softmax'))
oasis_model.summary()

oasis_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy', 'MeanSquaredError', 'AUC'])
oasis_train_dropout = oasis_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))