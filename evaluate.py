from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import random
from operator import truediv
from MSANS import MSAN
# import imageio
from model_DFFN import DFFNnet
import time
from model_DenseNet import densenet
# from ShufflleNetV2 import ShuffleNet
from model_LAnet import lanet
from model_HybridSN import HybridSN
# 加载数据
def loadData(names):
    if names == 'IP':
        data = sio.loadmat('datasets/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('datasets/Indian_pines_gt.mat')['indian_pines_gt']
    elif names == 'PU':
        data = sio.loadmat('./datasets/PaviaU.mat')['paviaU']
        labels = sio.loadmat('./datasets/PaviaU_gt.mat')['paviaU_gt']
    elif names == 'SA':
        data = sio.loadmat('./datasets/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('./datasets/Salinas_gt.mat')['salinas_gt']
    return data, labels
def pad(X, margin=14):
    paddingdata = np.pad(X, ((margin, margin), (margin, margin), (0, 0)),
                         "constant")  # 采用边缘值填充 [203, 203, 200]
    return paddingdata

# pca降维
def pca_change(X, num_components):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX


# 补0
def padwithzeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]    #
    list_diag = np.diag(confusion_matrix)                        #获取confusion_matrix的主对角线所有数值
    list_raw_sum = np.sum(confusion_matrix, axis=1)              #将主对角线所有数求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))   #list_diag/list_raw_sum  对角线各个数字/对角线所有数字的总和
    average_acc = np.mean(each_acc)                              #AA=
    return each_acc, average_acc


x_final, y_final = loadData('IP')
X_test = np.load('test/X_test.npy')
Y_test = np.load('test/Y_test.npy')

x_final = pca_change(x_final, num_components=32)
height = y_final.shape[0]
width = y_final.shape[1]
bands = x_final.shape[2]
x_final = x_final.reshape((-1, bands))  # 压缩尺寸x.shape=(21025,200),方便进行归一化
transfer = MinMaxScaler()  # 进行归一化操作
x_final = transfer.fit_transform(x_final)
x_final = x_final.reshape((height, width, bands))  # 将尺寸还原成(145,145,200)

PATCH_SIZE = 27

bands = x_final.shape[2]
x_final = pad(x_final, PATCH_SIZE // 2)
nb_classes = y_final.max()
img_rows = img_cols = PATCH_SIZE
#导入模型
#model = DFFNnet(img_rows, img_cols,bands, nb_classes, model_summary=True)
model = MSAN(bands, img_rows, img_cols, nb_classes, falg_summary=True)
#model = ShuffleNet(bands, img_rows, img_cols, nb_classes, falg_summary=True)
#model = lanet(input_size=(img_rows, img_cols, bands), num_class=nb_classes, model_summary=True)
#model = HybridSN(img_rows, img_cols,bands, nb_classes, model_summary=True)
#model = densenet(input_size=(27,27,32), num_class=16, model_summary=True)
# 加载模型参数
model.load_weights('test/MSAN-31-0.6615.hdf5')


def reports(X_test, y_test):
    # start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    # end = time.time()
    # print(end - start)
    target_names = ['0', '1', '2', '3',
                    '4', '5', '6',
                    '7', '8', '9', '10', '11',
                    '12', '13', '14',
                    '15']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)  # 计算OA
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)  # 计算confusion
    each_acc, aa = AA_andEachClassAccuracy(confusion)  # 计算each_acc和aa
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)  # 计算kappa
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100

    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(X_test, Y_test)

# calculate the predicted image
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        print('i = :', i)
        print('j = :', j)
        target = int(y_final[i, j])
        if target == 0:
            continue
        else:
            image_patch = patch(x_final, i, j)
            X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                               image_patch.shape[2]).astype('float32')
            prediction = (model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction + 1
 # 计算准确率
start = time.perf_counter()
Y_pred = model.predict(X_test)
end =time.perf_counter()
print('infer_time:', end-start)
classification = classification_report(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1), )
print(classification)

acc = np.mean(np.equal(outputs, y_final))
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('acc: ', acc)

print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('oa: ', oa)
print('each_acc: ', each_acc)
print('aa: ', aa)
print('kappa: ', kappa)
# Y_pred = model.predict(X_test)
# classification = classification_report(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1),)
# print(classification)

predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(12, 12))
predict_image1 = spectral.imshow(classes=y_final.astype(int), figsize=(12, 12))
# outputs.save(r'D:\jiang\trainModel\shijie\fuxian\MSANs\image\MSAN-385-0.1918.jpg')
#imageio.imwrite(os.path.join('msan1/', 'MSAN'+'_acc-'+str(round(acc, 4))+'.tif'), outputs)

a=2
