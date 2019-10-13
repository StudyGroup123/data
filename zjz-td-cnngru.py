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
import time
data_name='suochi-4000'
#data_name='benchmark-10_时序'

#-------------------------
if data_name=='suochi-4000':
    timesteps = 4000
    #num_classes = 9
    data = np.loadtxt('/home/bridge3/lstnet/kreas/data/suoshi-ziz/suochi-zjz-4000-4000-data/suochi-zjz-4000-4000-data.txt')
    label = np.loadtxt('/home/bridge3/lstnet/kreas/data/suoshi-ziz/suochi-zjz-4000-4000-data/suochi-zjz-4000-4000-label.txt')
if data_name=='com':
    timesteps = 5
    #num_classes = 9
    data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/com/com-4/comdata-10.txt')
    label = np.loadtxt('/home/bridge3/lstnet/esncnn/data/com/com-4/comlabel.txt')
if data_name=='bookshelf-20':
    timesteps = 20
    num_classes = 10
    data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf_pin20fft40.txt')
    #data =  np.loadtxt('/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf-20-data.txt')
    label = np.loadtxt('/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf-20-label.txt')
if data_name=='benchmark-10':
    timesteps = 5
    #data = np.loadtxt('/home/bridge3/lstnet/esncnn/data/beili/beili-50/beili-50-data.txt')
    #label = np.loadtxt('/home/bridge3/lstnet/esncnn/data/beili/beili-50/beili-50-label.txt')

    data = np.loadtxt(r'E:\github_new\lstnet\esncnn\data\beili\beili-50\beili-50-data.txt')
    #data = np.loadtxt(r'E:\github_new\lstnet\esncnn\data\com\com-4\comdata.txt')
    #label = np.loadtxt(r'E:\github_new\lstnet\esncnn\data\com\com-4\comlabel.txt')

    label = np.loadtxt(r'E:\github_new\lstnet\esncnn\data\beili\beili-50\beili-50-label.txt')


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
x_train, x_test, y_train, y_test = train_test_split(data, label_one_hot, test_size=0.2,shuffle=True,)


#bantch=int(data.shape[0]/200)
bantch=30

#network# 1layer

input=Input(shape=(data.shape[1:]))
# 0layer
l_1 = data.shape[1]
cnn_list = [int(l_1 * 0.2), int(l_1 * 0.4), int(l_1 * 0.6), int(l_1 * 0.8)]
# cnn_list=[1,2,3,4]
x11 = Conv1D(16, kernel_size=cnn_list[0], strides=3, activation='relu' )(input)
# x11 = AveragePooling1D(pool_size=pool_size1)(x11)
# 第二层
x12 = Conv1D(16, kernel_size=cnn_list[1], strides=3, activation='relu')(input)
# x12 = AveragePooling1D(pool_size=pool_size1)(x12)
# 第三层
x13 = Conv1D(16, kernel_size=cnn_list[2], strides=3, activation='relu')(input)
x14 = Conv1D(16, kernel_size=cnn_list[3], strides=3, activation='relu')(input)
g11 = GlobalAveragePooling1D()(x11)
g12 = GlobalAveragePooling1D()(x12)
g13 = GlobalAveragePooling1D()(x13)
g14 = GlobalAveragePooling1D()(x14)
#-------------------------
lstm=Bidirectional(LSTM(32,return_sequences=True))(input)
lstm=Bidirectional(LSTM(32))(lstm)
body_feature = concatenate([g11, g12, g13, g14, lstm])

dense=Dense(128,activation='relu')(body_feature)
dense=Dense(64,activation='relu')(dense)
output=Dense(y_test.shape[1],activation='softmax')(dense)

model=Model(input=input,output=output)

model.summary()
print('loading:',data_name)
#optimizer = SGD(lr=0.001, momentum=0.9) #re->0.90
optimizer = Adam(lr=0.01) #re->0.90
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

if os.path.isdir('./'+data_name+'/')==False:
    os.makedirs('./'+data_name+'/')
#||||||||||||SAME|||||||||||||||
callbacks=[
    EarlyStopping(monitor='val_accuracy', patience=300, verbose=1, mode='auto'),
    ModelCheckpoint('./'+data_name+'/'+data_name+'_model'+r+'.h5',monitor = 'val_accuracy',save_best_only=True),
]
history=model.fit(x_train,y_train,batch_size=bantch,epochs=100,validation_split=0.25,callbacks=callbacks)
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
plt.plot(np.arange(len(history.history['accuracy'])),history.history['accuracy'])
plt.plot(np.arange(len(history.history['val_accuracy'])),history.history['val_accuracy'])
np.savetxt('./'+data_name+'/'+data_name+'_val_acc'+r+'.txt',history.history['val_accuracy'])
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