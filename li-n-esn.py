# -*- coding:utf-8 -*-
from ucr_reading import *
#--------parameter adjustment=_path##############################
if "Windows" in platform.system():
    #_path='E:\\github_new\\lstnet\\esncnn\\data\\ECGME\\ECGME-32'
    _path='E:\github_new\lstnet\esncnn\data\com\com-4'
else:
    #_path='/home/bridge3/lstnet/esncnn/data/bookshelf/bookshelf-30'
    #_path='/home/bridge3/lstnet/esncnn/data/benchmark/benchmark-20'
    _path = '/home/bridge3/lstnet/esncnn/data/com/com-4'
##############################################################
if "Windows" in platform.system():
    listpath=_path.split('\\')
else:
    listpath = _path.split('/')
_listpath=listpath[-1].split("-")
dataName=_listpath[0]
shi=int(_listpath[1])
shi=20
print('Loading data...',dataName)
#--------parameter adjustment=###########################
number_res = 64 # 储备池大小
IS = 10   # 尺度因子
SR = 0.7    # 谱半径 #注意：不能大于10要不然参数错我
SP = 0.4    # 稀疏程度 #注意：不能大于10要不然参数错我
################################################################

train_echoes, train_y, dataset_name, n_res, IS, SR, SP= run_loading(dir_path=_path, n_res =number_res,IS=IS,SR=SR,SP=SP,timestep=shi)
#train_echoes, train_y, dataset_name, n_res, IS, SR, SP
#save
allData = train_echoes.reshape(train_echoes.shape[0], -1)
if "Windows" in platform.system():
    #print(_path+"\\"+dataName+"_shi"+str(shi)+"is"+str(IS)+"sr"+str(int(SR*10))+"sp"+str(int(SP*10))+".txt")
    np.savetxt(_path+"\\"+dataName+"_shi"+str(shi)+"is"+str(IS)+"sr"+str(int(SR*10))+"sp"+str(int(SP*10))+".txt",allData,fmt='%.6e')
else:
    np.savetxt(_path+"/"+dataName+"_shi"+str(shi)+"is"+str(IS)+"sr"+str(SR*10)+"sp"+str(SP*10)+".txt",allData,fmt='%.6e')
print(_path+"\\"+dataName+"_shi"+str(shi)+"is"+str(IS)+"sr"+str(int(SR*10))+"sp"+str(int(SP*10))+".txt","shiYu---shiYu  save finally!!!!!")