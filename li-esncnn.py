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
import re
mpl.use('Agg')
from matplotlib import pyplot as plt
import os
import random
import platform

r=str(int(float(time.time())))

def multi_input_model():
	"""构建多输入模型"""

	input1_ = Input(shape=(x1.shape[1],x1.shape[2]), name='input1')
	input2_ = Input(shape=(x2.shape[1],x2.shape[2]), name='input2')
# 时域的特征 窗口小
	#第一层
	l_1=x1.shape[1]
	cnn_list=[int(l_1*0.2),int(l_1*0.2),int(l_1*0.6),int(l_1*0.8)]
	#cnn_list=[1,2,3,4]
	x11 = Conv1D(32, kernel_size=cnn_list[0], strides=1, activation='relu', padding='same')(input1_)
	#x11 = AveragePooling1D(pool_size=pool_size1)(x11)
	# 第二层
	x12 = Conv1D(32, kernel_size=cnn_list[1], strides=1, activation='relu', padding='same')(input1_)
	#x12 = AveragePooling1D(pool_size=pool_size1)(x12)
	# 第三层
	x13 = Conv1D(32, kernel_size=cnn_list[2], strides=1, activation='relu', padding='same')(input1_)
	x14 = Conv1D(32, kernel_size=cnn_list[3], strides=1, activation='relu', padding='same')(input1_)
	g11=GlobalAveragePooling1D()(x11)
	g12=GlobalAveragePooling1D()(x12)
	g13=GlobalAveragePooling1D()(x13)
	g14=GlobalAveragePooling1D()(x14)
	#x13 = AveragePooling1D(pool_size=pool_size1)(x13)
# 频域的特征 窗口大
	l_2 = x2.shape[1]
	lstm_list = [int(l_2 * 0.2), int(l_2 * 0.4), int(l_2 * 0.6),int(l_2 * 0.8)]
	#lstm_list = [10,50,100,200]
	x21 = Conv1D(32, kernel_size=lstm_list[0], strides=1, activation='relu', padding='same')(input2_)
	#x21 = AveragePooling1D(pool_size=pool_size2)(x21)
	# 第二层
	x22 = Conv1D(32, kernel_size=lstm_list[1], strides=1, activation='relu', padding='same')(input2_)
	#x22 = AveragePooling1D(pool_size=pool_size2)(x22)
	# 第三层
	x23 = Conv1D(32, kernel_size=lstm_list[2], strides=1, activation='relu', padding='same')(input2_)

	x24 = Conv1D(32, kernel_size=lstm_list[3], strides=1, activation='relu', padding='same')(input2_)
	#x23 = AveragePooling1D(pool_size=pool_size2)(x23)
	g21=GlobalAveragePooling1D()(x21)
	g22=GlobalAveragePooling1D()(x22)
	g23=GlobalAveragePooling1D()(x23)
	g24=GlobalAveragePooling1D()(x24)
	body_feature=concatenate([g11,g12,g13,g14,g21,g22,g23,g24])
	body_feature = Dropout(0.2)(body_feature)
	#x = concatenate([x1, x2])
	#特征融合
	#x = concatenate([x11, x12,x13,x21, x22,x23])
	# y = concatenate([])
	# xy=concatenate([x,y])
	#x = Flatten()(x)
	# 特征展评
	body_feature = Dense(64, activation='relu')(body_feature)
	body_feature = Dense(32, activation='relu')(body_feature)
	#body_feature = Dense(128, activation='relu')(body_feature)
	output_ = Dense(num_class, activation='softmax', name='output')(body_feature)
	model = Model(inputs=[input1_, input2_], outputs=[output_])
	model.summary()
	return model


if __name__ == '__main__':
	# 产生训练数据 注意：参数调整需要修改的
	list =4
	#----------------------------------------------------------------从这个地方开始，下面参数不需要修改
	if list==4:
		data_name = 'com-4'
		if "Windows" in platform.system():
			pathx1 = 'E:\github_new\lstnet\esncnn\data\com\com-4\com_shi4is20sr7.0sp4.0.txt'  # 时域
			pathx2 = 'E:\github_new\lstnet\esncnn\data\com\com-4\com_pin4fft100.txt'  # 频域
			label  = 'E:\github_new\lstnet\esncnn\data\com\com-4\comlabel.txt'
		else:
			pathx1 = '/home/bridge3/lstnet/esncnn/data/com/com-4/com_shi4is20sr7.0sp4.0.txt'  # 时域
			pathx2 = '/home/bridge3/lstnet/esncnn/data/com/com-4/com_pin4fft100.txt'  # 频域
			label  = '/home/bridge3/lstnet/esncnn/data/com/com-4/comlabel.txt'
		time_shi = 4 #步长
		time_pin = 100 #fft的长度
		epochs=2000
	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------从这个地方开始，下面参数不需要修改
	if list==3:
		data_name = '0.3kg-20'
		if "Windows" in platform.system():
			pathx1 = 'wwwwE:\\github_new\\lstnet\\esncnn\\data\\0.3kg\\0.3kg-20\\0.3kg_shi20is20sr7.0sp4.0.txt'  # 时域
			# pathx2 = 'E:\\github_new\\lstnet\\esncnn\\data\\0.3kg\\0.3kg-20\\0.3kg_pin20fft280.txt'  # 频域
			# label  = 'E:\\github_new\\lstnet\\esncnn\\data\\0.3kg\\0.3kg-20\\0.3kg-20-label.txt'
		else:
			pathx1 = '/home/bridge3/lstnet/esncnn/data/0.3kg/0.3kg-20/0.3kg_shi20is20sr7.0sp4.0.txt'  # 时域
			pathx2 = '/home/bridge3/lstnet/esncnn/data/0.3kg/0.3kg-20/0.3kg_pin20fft40.txt'  # 频域
			label  = '/home/bridge3/lstnet/esncnn/data/0.3kg/0.3kg-20/0.3kg-20-label.txt'
		time_shi = 20 #步长
		time_pin = 40 #fft的长度
		epochs=200
	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------从这个地方开始，下面参数不需要修改
	if list==2:
		data_name = 'bookshelf-20'
		if "Windows" in platform.system():
			pathx1 = 'wwwwwE:\\github_new\\lstnet\\esncnn\\data\\0.15kg\\0.15kg-20\\0.15kg_shi20is20sr7.0sp4.0.txt'  # 时域
			# pathx2 = 'E:\\github_new\\lstnet\\esncnn\\data\\0.15kg\\0.15kg-20\\0.15kg_pin20fft280.txt'  # 频域
			# label  = 'E:\\github_new\\lstnet\\esncnn\\data\\0.15kg\\0.15kg-20\\0.15kg-20-label.txt'
		else:
			pathx1 = '/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf_shi20is20sr7.0sp4.0.txt'  # 时域
			pathx2 = '/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf_pin20fft40.txt'  # 频域
			label = '/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20/bookshelf-20-label.txt'
		time_shi = 20 #步长
		time_pin = 40 #fft的长度
		epochs=100
	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------从这个地方开始，下面参数不需要修改
	if list==1:
		data_name = 'benchmark-10'
		if "Windows" in platform.system():
			pathx1 = 'wwwwE:\\github_new\\lstnet\\esncnn\\data\\ECGME\\ECGME-32\\benchmark_shi20is20sr7.0sp4.0.txt'  # 时域
			pathx2 = 'wwwE:\\github_new\\lstnet\\esncnn\\data\\ECGME\\ECGME-32\\ECGME_pin32fft256.txt'  # 频域
			label = 'E:\\github_new\\lstnet\\esncnn\\data\\ECGME\\ECGME-32\\ECGme_label.txt'
		else:
			pathx1 = '/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-10/benchmark_shi10is20sr7sp4.txt'  # 时域
			pathx2 = '/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-10/benchmark_pin10fft20.txt'  # 频域
			label = '/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-10/benchmark-10-label.txt'
		time_shi = 10#步长
		time_pin = 20#fft的长度
		epochs=100
	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------从这个地方开始，下面参数不需要修改
	if list==0:
		data_name = 'ECGME-32'
		if "Windows" in platform.system():
			pathx1 = 'E:\\github_new\\lstnet\\esncnn\\data\\ECGME\\ECGME-32\\ECGME_shi32is20sr7sp4.txt'  # 时域
			pathx2 = 'E:\\github_new\\lstnet\\esncnn\\data\\ECGME\\ECGME-32\\ECGME_pin32fft256.txt'  # 频域
			label = 'E:\\github_new\\lstnet\\esncnn\\data\\ECGME\\ECGME-32\\ECGme_label.txt'
		else:
			pathx1 = '/home/bridge3/lstnet/esncnn/data/ECGME/ECGME-32/ECGME_shi32is20sr7sp4.txt'  # 时域
			pathx2 = '/home/bridge3/lstnet/esncnn/data/ECGME/ECGME-32/ECGME_pin32fft256.txt'  # 频域
			label = '/home/bridge3/lstnet/esncnn/data/ECGME/ECGME-32/ECGme_label.txt'
		time_shi = 32#步长
		time_pin = 256#fft的长度
	#----------------------------------------------------------------------------------------

	if time_pin%time_shi==0:
		pool_size1=2
		pool_size2=int(time_pin/time_shi*2)
		print(data_name+'loading success!-》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》!')
	else:
		print("程序退出，频域与时间步没有是倍数关系")
		exit()
	#-载入数据,数据的装换
	_x1 = np.loadtxt(pathx1) #时域
	_x2 = np.loadtxt(pathx2)#频域
	_y = np.loadtxt(label) #标签

	# -载入数据,数据的按比例划分
	_x1_train, _x1_test, y1_train, y1_test = train_test_split(_x1, _y, random_state=100, test_size=0.00001,shuffle=True, )
	_x2_train, _x2_test, y2_train, y2_test = train_test_split(_x2, _y, random_state=100, test_size=0.00001,shuffle=True, )
	#形成训练测试数据与类数
	#注意y1_train=y2_train；y1_test=y2_test
	y1_train = np_utils.to_categorical(y1_train)
	y1_test = np_utils.to_categorical(y1_test)
	num_class=y1_train.shape[1]
	#训练数据与测试数据进行维度变换
	x1_train=_x1_train.reshape(y1_train.shape[0],time_shi,-1)
	x2_train=_x2_train.reshape(y1_train.shape[0],time_pin,-1)
	#-----
	x1_test=_x1_test.reshape(y1_test.shape[0],time_shi,-1)
	x2_test=_x2_test.reshape(y1_test.shape[0],time_pin,-1)
	#--------------
	x1=x1_train
	x2=x2_train
	y=y1_train
	model = multi_input_model()
	#开始训练模型了
	if os.path.isdir('./' + data_name + '/') == False:
		os.makedirs('./' + data_name + '/')

	callbacks = [
		EarlyStopping(monitor='val_acc', patience=300, verbose=1, mode='auto'),
		# ModelCheckpoint('./' + data_name + '/' + data_name + '_model' + r + '.h5', monitor='val_acc',
		#                 save_best_only=True),
	]
	optimizer = Adam(lr=0.0001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
	history=model.fit([x1, x2], y, epochs=epochs,validation_split=0.1,callbacks=callbacks,shuffle=True)
	#---测试集的混淆矩阵、评价指标、loss
	score_test = model.evaluate([x1_test, x2_test],y1_test)
	pre_test = model.predict([x1_test, x2_test])
	print(confusion_matrix(np.argmax(pre_test, axis=1).flatten(), np.argmax(y1_test, axis=1).flatten()))
	print(classification_report(np.argmax(pre_test, axis=1).flatten(), np.argmax(y1_test, axis=1).flatten()))
	print("test loss:", score_test[0])
	print(">>>>>>>>>>>>>>>>[test_accuracy]", score_test[1])
	#---训练集的混淆矩阵、评价指标、loss
	score_train = model.evaluate([x1, x2], y1_train)
	print("train loss:", score_train[0])
	print(">>>>>>>>>>>>>>>>[train_accuracy]", score_train[1])
	pre_train = model.predict([x1, x2])
	print(confusion_matrix(np.argmax(pre_train, axis=1).flatten(), np.argmax(y1_train, axis=1).flatten()))
	print(classification_report(np.argmax(pre_train, axis=1).flatten(), np.argmax(y1_train, axis=1).flatten()))

#-------画图以及保存画图的数据
	plt.figure()
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.plot(np.arange(len(history.history['acc'])), history.history['acc'])
	plt.plot(np.arange(len(history.history['val_acc'])), history.history['val_acc'])
	np.savetxt('./' + data_name + '/' + data_name + '_val_acc' + r + '.txt', history.history['val_acc'])
	plt.legend(['Training', 'Validation'], loc='lower right')
	plt.savefig('./' + data_name + '/' + data_name + r + '_dnn_acc.png')
	plt.clf()

	plt.figure()
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.plot(np.arange(len(history.history['loss'])), history.history['loss'])
	plt.plot(np.arange(len(history.history['val_loss'])), history.history['val_loss'])
	np.savetxt('./' + data_name + '/' + data_name + '_val_loss' + r + '.txt', history.history['val_loss'])
	plt.legend(['Training', 'Validation'])
	plt.savefig('./' + data_name + '/' + data_name + r + '_dnn_loss.png')
	plt.clf()