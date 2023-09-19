#coding:utf-8

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
# from model_LAnet import lanet
# from SWCNet import MSAN
from sklearn.preprocessing import MinMaxScaler
# #from MSANS import MSAN
# from MADANet import MSAN
# from model_LAnet import lanet
# from model_DFFN import DFFNnet, build_discriminator 
# from model_DenseNet import densenet
# from SW_CNN import SW_CNN
from A2MFE import MFE

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


def pixel_select(X, Y):
    test_pixels = Y.copy()  # 复制Y到test_pixels
    kinds = np.unique(Y).shape[0] - 1  # np.unique(Y)=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],dtype=uint8) ,kinds=分类种类数
    # print(kinds)
    for i in range(kinds):  # i从0-15
        num = np.sum(Y == (i + 1))  # 计算每个类总共有多少样本 ,从Y=1到Y=16
        #        print(num)
        # # print(num)
        # train_num = [14, 419, 265, 69, 149, 219, 9, 144, 5, 288, 733, 175, 65, 376, 119, 31]  # %30的样本
        train_num = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # %10的样本
#         train_num = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        temp1 = np.where(Y == (i + 1))  # 返回标签满足第i+1类的位置索引，第一次循环返回第一类的索引
        temp2 = random.sample(range(num), train_num[
            i])  # get random sequence,random.sample表示从某一序列中随机获取所需个数（train_num）的数并以片段的形式输出,,再这里将随机从每个种类中挑选train_num个样本
        for i in temp2:
            test_pixels[temp1[0][temp2], temp1[1][temp2]] = 0  # 除去训练集样本
    train_pixels = Y - test_pixels
    return train_pixels, test_pixels


# Get the cubes
def GetImageCubes(input_data, pixels_select):                         #这里的label_select就是train_pixels/test_pixels
    Band = input_data.shape[2]
    kind = np.unique(pixels_select).shape[0]-1                     #得到测试或者训练集中的种类数
    patch = 28
    #print(kind)
    paddingdata = np.pad(input_data, ((15, 15), (15, 15), (0, 0)), "constant")  # 采用边缘值填充 [203, 203, 200]                 可以作为超参数
    paddinglabel = np.pad(pixels_select, ((15, 15), (15, 15) ), "constant")
    #得到 label的 pixel坐标位置,去除背景元素
    pixel = np.where(paddinglabel != 0)         # pixel = np.where(label_select != 0)  ，这里的pixel是坐标数据，不是光谱数据
    #the number of batch
    num = np.sum(pixels_select != 0)             # 参与分类的像素点个数
    batch_out = np.zeros([num, patch, patch, Band])
    batch_label = np.zeros([num, kind])
    for i in range(num):                         # 得到每个像素点的batch，在这里为19*19的方块
        row_start = pixel[0][i] - (patch // 2)
        row_end = pixel[0][i] + (patch // 2 )
        col_start = pixel[1][i] - (patch // 2)
        col_end = pixel[1][i] + (patch // 2 )
        batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :]      # 得到一个数据块
        temp = (paddinglabel[pixel[0][i], pixel[1][i]]-1)    #temp = (label_selct[pixel[0][i],pixel[1][i]]-1)
        batch_label[i, temp] = 1                             # 独热编码，并且是从零开始的
    #修改合适三维卷积输入维度 [depth height weight]
    #batch_out = batch_out.swapaxes(1, 3)
    #batch_out = batch_out[:, :, :, :, np.newaxis]           # np.newaxis:增加维度
    return batch_out, batch_label


def padwithzeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def creatCube(X, y, windowsize=25, removeZeroLabels=True):
    margin = int((windowsize - 1) / 2)  # margin=12
    zeroPaddedX = padwithzeros(X, margin=margin)

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowsize, windowsize, X.shape[2]))  # (145*145,25,25,30)
    patchesLabels = np.zeros(X.shape[0] * X.shape[1])
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):  # (12,145-12)=(12,132)
        for c in range(margin, zeroPaddedX.shape[1] - margin):  # (12,145-12)=(12,132)
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels
# print(history.history.keys())
# pca降维
def pca_change(X, num_components):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX

X, Y = loadData('IP')
nb_classes = Y.max()

X = pca_change(X, num_components=32)

[h, w, bands] = X.shape
nb_classes = Y.max()
X = X.reshape((-1, bands))  # 压缩尺寸x.shape=(21025,200),方便进行归一化
transfer = MinMaxScaler()  # 进行归一化操作
X = transfer.fit_transform(X)
X = X.reshape((h, w, bands))  # 将尺寸还原成(145,145,200)

train_pixels, test_pixels = pixel_select(X, Y)


X_train, Y_train = GetImageCubes(X, train_pixels)# 256 *27*27*32
X_test, Y_test = GetImageCubes(X, test_pixels) 
# X_, Y_ = creatCube(X, Y, windowsize=27)
# Y_ = np_utils.to_categorical(Y_)


def splitTrainTest(X, y, Ratio, randoms=2019):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=Ratio, random_state=randoms,
                                                        stratify=y)
    return X_train, X_test, Y_train, Y_test

patch = 28
# # 划分训练集和测试集
# X_train, X_test, Y_train, Y_test = splitTrainTest(X_, Y_, Ratio=0.2)
# # 划分训练集和验证集
X_test, X_val, Y_test, Y_val = splitTrainTest(X_test, Y_test, Ratio=0.2)
# np.save('test1/X_test.npy', X_test)
# np.save('test1/Y_test.npy', Y_test)
# np.save('test1/X_train.npy', X_train)
# np.save('test1/Y_train.npy', Y_train)
save_dir = os.path.join(os.getcwd(), 'MFE')
save_log = os.path.join(os.getcwd(), 'log')
save_model_path = os.path.join(save_dir, 'MFE' + '-{epoch:02d}-{val_accuracy:.4f}.hdf5')
# filepath = 'MSAN.hdf5'
checkpoint = ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                             mode='auto', save_weights_only=True)
tensorboard = TensorBoard(log_dir=save_log, histogram_freq=0, write_graph=True, write_images=True)
callback_list = [checkpoint, tensorboard]

img_rows = img_cols = patch
model = MFE(bands, img_rows, img_cols, nb_classes, falg_summary=True)
#model = DFFNnet(img_rows, img_cols,bands, nb_classes, model_summary=True)
#model = build_discriminator(img_rows, img_cols,bands, nb_classes, model_summary=True)
# model = unet(input_size=(img_rows, img_cols, bands), classes = nb_classes)
#model = lanet(input_size=(img_rows, img_cols, bands), num_class=nb_classes, model_summary=True)
#model = densenet(input_size=(25,25,32), num_class=16, model_summary=True)
#model = SW_CNN(bands, img_rows, img_cols, nb_classes, falg_summary=True, pretrained_weights=None, model_plot=False)
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=500, batch_size=32, verbose=1, shuffle=True, callbacks=callback_list)
#with open(save_log + '/mul.txt', 'w') as f:
 #   f.write(str(history.history))

# 加载模型参数
# model.load_weights('my_model_weights.hdf5')


start = time.perf_counter()
Y_pred = model.predict(X_test)
end =time.perf_counter()
print('infer_time:', end-start)
classification = classification_report(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1),)
print(classification)


#打印epochs的损失和精度
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'])
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'])
plt.show()
