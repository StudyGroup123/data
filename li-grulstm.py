# -*- coding:utf-8 -*-
# 1 lodaing lib
import numpy as np
from keras.models import *
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import  *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *
from keras.callbacks import *
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os
import random

#data_name='bookshelf-20'22wss12222
data_name='suochi_0.3_20'
#data_name='benchmark-10_时序'

#-------------------------
if data_name=='bookshelf-20':
    timesteps = 20
    num_classes = 10
    data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf_pin20fft40.txt')
    #data =  np.loadtxt('/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf-20-data.txt')
    label = np.loadtxt('/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf-20-label.txt')
if data_name=='benchmark-10':
    timesteps = 10
    num_classes = 9
    data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-10/benchmark_pin10fft20.txt')
    label = np.loadtxt('/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-10/benchmark-10-label.txt')
if data_name=='suochi_0.3_20':
    timesteps = 20  #时序是20频域是40
    num_classes = 4
    #data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/0.3kg/0.3kg-20/0.3kg_pin20fft40.txt')
    data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/0.3kg/0.3kg-20/0.3kg-20-data.txt')
    label =np.loadtxt('/home/bridge3/lstnet/esncnn/data/0.3kg/0.3kg-20/0.3kg-20-label.txt')
if data_name=='benchmark-10_时序':
    timesteps = 10
    num_classes = 9
    data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-10/benchmark-10-data.txt')
    label =np.loadtxt('/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-10/benchmark-10-label.txt')

# data reshape
r=str(int(float(time.time())))
data = data.reshape(-1, timesteps, int(data.shape[1]/timesteps))
label_one_hot = np_utils.to_categorical(label)
x_train, x_test, y_train, y_test = train_test_split(data, label_one_hot, random_state=100, test_size=0.2,shuffle=True,)


#bantch=int(data.shape[0]/200)
bantch=512

#network# 1layer22
input=Input(shape=(data.shape[1:]))
#ip = Input(shape=(100, MAX_SEQUENCE_LENGTH))

x = Bidirectional(GRU(10,return_sequences=True))(input)
x = Bidirectional(GRU(10))(x)
x = Dropout(0.2)(x)

#y = Permute((2, 1))(ip)
y1 = Conv1D(64, 10, kernel_initializer='he_uniform',activation='relu')(input)
y2 = Conv1D(64, 20, kernel_initializer='he_uniform', activation='relu')(input)
#y = BatchNormalization()(y)
#y = Activation('relu')(y)
y1=Dropout(0.2)(y1)
y2=Dropout(0.2)(y2)
y1 = GlobalAveragePooling1D()(y1)
y2 = GlobalAveragePooling1D()(y2)
y = concatenate([y1, y2])

x = concatenate([x, y])
x = Dense(512)(x)
x = Dense(256)(x)
x = Dense(128)(x)
output=Dense(y_test.shape[1],activation='softmax')(x)
model=Model(input=input,output=output)
model.summary()
print('loading:',data_name)
#optimizer = SGD(lr=0.001, momentum=0.9) #re->0.90
optimizer = Adam(lr=0.001) #re->0.90
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

if os.path.isdir('./'+data_name+'/')==False:
    os.makedirs('./'+data_name+'/')
#||||||||||||SAME|||||||||||||||
callbacks=[
    EarlyStopping(monitor='val_acc', patience=300, verbose=1, mode='auto'),
    ModelCheckpoint('./'+data_name+'/'+data_name+'_model'+r+'.h5',monitor = 'val_acc',save_best_only=True),
]
history=model.fit(x_train,y_train,batch_size=bantch,epochs=500,validation_split=0.3,callbacks=callbacks,shuffle=True)
model = load_model('./'+data_name+'/'+data_name+'_model'+r+'.h5')

score_test=model.evaluate(x_test,y_test)
pre_test=model.predict(x_test)
print(confusion_matrix(np.argmax(pre_test,axis=1).flatten(),np.argmax(y_test,axis=1).flatten()))
print(classification_report(np.argmax(pre_test,axis=1).flatten(),np.argmax(y_test,axis=1).flatten()))
print("test loss:",score_test[0])
print(">>>>>>>>>>>>>>>>[test_accuracy]",score_test[1])
score_train=model.evaluate(x_train,y_train)
print("train loss:",score_train[0])
print(">>>>>>>>>>>>>>>>[train_accuracy]",score_train[1])
pre_train=model.predict(x_train)
print(confusion_matrix(np.argmax(pre_train,axis=1).flatten(),np.argmax(y_train,axis=1).flatten()))
print(classification_report(np.argmax(pre_train,axis=1).flatten(),np.argmax(y_train,axis=1).flatten()))
#plt accuracy loss
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(np.arange(len(history.history['acc'])),history.history['acc'])
plt.plot(np.arange(len(history.history['val_acc'])),history.history['val_acc'])
np.savetxt('./'+data_name+'/'+data_name+'_val_acc'+r+'.txt',history.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')
plt.savefig('./'+data_name+'/'+data_name+r+'_acc.png')
plt.clf()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(np.arange(len(history.history['loss'])),history.history['loss'])
plt.plot(np.arange(len(history.history['val_loss'])),history.history['val_loss'])
np.savetxt('./'+data_name+'/'+data_name+'_val_loss'+r+'.txt', history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.savefig('./'+data_name+'/'+data_name+r+'_loss.png')
plt.clf()