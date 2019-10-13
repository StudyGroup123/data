import platform
import os
import numpy as np

# --------parameter adjustment=_path####################################
if "Windows" in platform.system():
	#_path = 'E:\\github_new\\lstnet\\esncnn\\data\\benchmark\\benchmark-20'
	_path = 'E:\github_new\lstnet\esncnn\data\com\com-4'
else:
	#_path = '/home/bridge3/lstnet/esncnn/data/0.3kg/0.3kg-20'
	_path = '/home/bridge3/lstnet/esncnn/data/com/com-4'
	#_path = '/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-20'
##################################################################
if "Windows" in platform.system():
	listpath = _path.split('\\')
else:
	listpath = _path.split('/')
_listpath = listpath[-1].split("-")
dataName = _listpath[0]
pin = int(_listpath[1])
print('Loading data...', dataName)
# --------parameter adjustment=###############################
fft = 100
######################################################
_listdir = os.listdir(_path)
for i in range(len(_listdir)):
	if "data" in _listdir[i]:
		_data = _listdir[i]

# read datass
if "Windows" in platform.system():
	print(_path + "\\" + _data)
	data = np.loadtxt(_path + "\\" + _data)
else:
	data = np.loadtxt(_path + "/" + _data)
# read data
newData = data.reshape(data.shape[0], pin, -1)
pinData = np.zeros((newData.shape[0], fft, newData.shape[2]))
for n_pin in range(pinData.shape[0]):
	for j_sensor in range(pinData.shape[2]):
		pinData[n_pin, :, j_sensor] = np.fft.fft(newData[n_pin, :, j_sensor], n=fft)
		print(str(n_pin) + " finally!")
allData = pinData.reshape(pinData.shape[0], -1)
if "Windows" in platform.system():
	np.savetxt(_path + "\\" + dataName + "_pin" + str(pin) + "fft" + str(fft) + ".txt", allData, fmt='%.6e')
else:
	np.savetxt(_path + "/" + dataName + "_pin" + str(pin) + "fft" + str(fft) + ".txt", allData, fmt='%.6e')
print(_path + "\\" + dataName + "_pin" + str(pin) + "fft" + str(fft) + ".txt", ">>>>>pinyu  save finally!!!!!")
