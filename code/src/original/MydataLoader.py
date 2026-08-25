from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from tqdm import *

import cv2

# HECKTOR data root directory (platform-independent)
_HECKTOR_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)))


def normalize_std(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-6)  # 标准化

def normalize_max_min(img):
    Max = np.max(img)
    Min = np.min(img)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化


class MyDataset(Dataset):
    def __init__(self, is_train):

        self.input = []
        self.target = []
        self.mulfactor = []
        self.petnoise = []

        data = np.load('./dataSetT.npz')
        np.random.seed(42)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
        n_examples = len(data['sin'])
        # n_examples = 1000
        n_train = n_examples * 0.8  
        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        test_idx = list(set(range(0,n_examples))-set(train_idx))

        # 5.1 训练还是测试数据集
        if is_train:
            self.input = data['sin'][train_idx]
            self.target = data['pet'][train_idx]
            self.mulfactor = data['mulfactor'][train_idx]
            self.petnoise = data['petnoise'][train_idx]
            print('Train Len:',len(self.input))

        else:
            self.input = data['sin'][test_idx]
            self.target = data['pet'][test_idx]
            self.mulfactor = data['mulfactor'][test_idx]
            self.petnoise = data['petnoise'][test_idx]
            print('Test Len:',len(self.input))


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        # image = normalize_std(image)
        # image = normalize_max_min(image)

        # input = np.expand_dims(self.input[idx],0).repeat(3,axis=0)
        # target = np.expand_dims(self.target[idx],0).repeat(3,axis=0)
        input = self.input[idx]
        target = normalize_max_min(self.target[idx])
        mul_factor = self.mulfactor[idx]
        pet_noise = self.petnoise[idx]


        # input = torch.tensor(input).float()
        # target = torch.tensor(target).float()
        # mul_factor = torch.tensor(mul_factor).float()
        # pet_noise = torch.tensor(pet_noise).float()

        return input,target,mul_factor,pet_noise
    


class MyDatasetK(Dataset):
    def __init__(self, is_train):

        self.input = []
        self.target = []
        self.mulfactor = []
        self.petnoise = []

        # data = np.load('./dataSetK3-T-KEM-192.npz')
        # np.random.seed(42)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
        # n_examples = len(data['pet'])
        # # n_examples = 1000
        # n_train = n_examples * 0.8  
        # train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        # test_idx = list(set(range(0,n_examples))-set(train_idx))

        # 5.1 训练还是测试数据集
        if is_train:
            data = np.load('./dataSetK3-T-KEM-192-Train.npz')
            self.input = data['input']
            self.target = data['target']


        else:
            data = np.load('./dataSetK3-T-KEM-192-Test.npz')
            self.input = data['input']
            self.target = data['target']


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        input = self.input[idx]
        target = self.target[idx]

        for i in range(len(input)):
            # input[i] = normalize_std(input[i])
            input[i] = normalize_max_min(input[i])
        #     input[i] = np.clip(input[i],0,10) /10

        # print(np.max(input),np.min(input))
            

        # target = normalize_std(target)
        target = normalize_max_min(target)
        # target = np.clip(target,0,1000) / 1000
        # target = np.clip(target,0,1000) / 1000

        # new_image = np.zeros([3,256,256])
        # for j in range(3):
        #     new_image[j]  = cv2.resize(input[j], (256, 256), interpolation=cv2.INTER_LINEAR)
        # target = normalize_max_min(self.target[idx])

        input = torch.tensor(input).float()
        target = torch.tensor(target).float()
        target = target.unsqueeze(0)

        return input,target
    

class MyDatasetB(Dataset):
    def __init__(self, is_train):


        # data = np.load('datasetB3.npz')
        # n_examples = len(data['pet'])
        # n_examples = 1000
        # n_train = int(n_examples * 0.8)  
 

        # 5.1 训练还是测试数据集
        if is_train:
            data = np.load('./dataSetBN-Train.npz')
            self.target = data['target']
            self.input = data['input']


        else:
            data = np.load('./dataSetBN-Test.npz')
            self.target = data['target']
            self.input = data['input']

        print(self.input.shape)


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        input = self.input[idx]
        target = self.target[idx]

        # target = pad_and_center_image(target,(160,160))

        for i in range(len(input)):
            input[i] = normalize_max_min(input[i])


        # print(np.max(input),np.min(input))
            

        # target = normalize_std(target)

        target = normalize_max_min(target)
        pet_gt_t = target
        # np.maximum(pet_gt_t, 0, out=pet_gt_t)
        # xmax = np.amax(pet_gt_t,axis=(-1,-2),keepdims=True)
        # xmax_tmp = np.ones_like(xmax)
        # xmax_tmp[xmax<0.1]=0.01
        # xmax_tmp[xmax<1]=0.1
        # xmax_tmp[xmax>10]=10
        # xmax_tmp[xmax>100]=100
        # xmax_tmp[xmax>1000]=1000
        # xmax_tmp[xmax>10000]=10000
        # xmax_tmp[xmax>1000000]=1000000
        # xmax_tmp[xmax>10000000]=10000000
        # xmax_tmp[xmax>100000000]=100000000
        # xmax_tmp[xmax>1000000000]=1000000000
        # xmax_tmp[xmax>10000000000]=10000000000
        # xmax = xmax_tmp
        # pet_gt_t = pet_gt_t/xmax

        # target = np.clip(target,0,1000) / 1000
        # target = np.clip(target,0,1000) / 1000

        # new_image = np.zeros([3,256,256])
        # for j in range(3):
        #     new_image[j]  = cv2.resize(input[j], (256, 256), interpolation=cv2.INTER_LINEAR)
        # target = normalize_max_min(self.target[idx])

        input = torch.tensor(input).float()
        target = torch.tensor(pet_gt_t).float()
        # input = input.unsqueeze(0)
        target = target.unsqueeze(0)

        return input,target
def normalize(I):
    if isinstance(I,np.ndarray):
        normI = (I-np.amin(I,axis=(-1,-2)))/(np.amax(I,axis=(-1,-2))-np.amin(I,axis=(-1,-2)))
    elif isinstance(I,torch.Tensor):
        normI = (I-torch.amin(I,dim=(-1,-2)))/(torch.amax(I,dim=(-1,-2))-torch.amin(I,dim=(-1,-2)))
    else:
        raise TypeError("I must be cupy.ndarray or numpy.ndarray or torch.Tensor")
    return normI
import random
import scipy

def pad_and_center_image(images, size):
    h, w = images.shape[-2:]
    modified_size = max(*size, h, w)
    dh = modified_size - h
    dw = modified_size - w

    # 计算补零的位置
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    # 补零
    assert top >= 0 and bottom >= 0 and left >= 0 and right >= 0, 'dh={}, dw={}, h={}, w={}'.format(dh, dw, h, w)
    if len(images.shape)==3:
        padded_image = np.zeros((images.shape[0],modified_size,modified_size))
        padded_image = padded_image+images[:,0:1,0:1]
        if top>=0 and bottom>0 and left>=0 and right>0:
            padded_image[:,top:-bottom,left:-right] = images
        elif top==0 and bottom==0 and left>=0 and right>0:
            padded_image[:,:,left:-right] = images
        elif left==0 and right==0 and top>=0 and bottom>0:
            padded_image[:,top:-bottom,:] = images
        elif top==0 and bottom==0 and left==0 and right==0:
            padded_image = images
        total_imgs = padded_image
        total_imgs = scipy.ndimage.zoom(total_imgs, (1,(size[0])/modified_size,(size[1])/modified_size),order=1)
    elif len(images.shape)==2:
        padded_image = np.zeros((modified_size,modified_size))
        padded_image = padded_image+images[0:1,0:1]
        if top>=0 and bottom>0 and left>=0 and right>0:
            padded_image[top:-bottom,left:-right] = images
        elif top==0 and bottom==0 and left>=0 and right>0:
            padded_image[:,left:-right] = images
        elif left==0 and right==0 and top>=0 and bottom>0:
            padded_image[top:-bottom,:] = images
        elif top==0 and bottom==0 and left==0 and right==0:
            padded_image = images
        total_imgs = padded_image
        total_imgs = scipy.ndimage.zoom(total_imgs, ((size[0])/modified_size,(size[1])/modified_size),order=1)

    assert total_imgs.shape[-2:] == size, 'total_imgs.shape={}, size={}'.format(total_imgs.shape, size)
    return total_imgs

class MyDatasetRawB(Dataset):
    def __init__(self, is_train):

        self.target = []


        data = np.load('D:/MIAProject/brainweb/pet_vol_slice_144by144_nt.npy')
        train_id = np.load('./train_list1.npy')
        test_id = np.load('./test_list1.npy')
        # n_examples = len(data)
        # # n_examples = 1000
        # n_train = int(n_examples * 0.8)  
 

        # 5.1 训练还是测试数据集
        if is_train:
            self.target = data[train_id]
            self.ct = data[train_id]


        else:
            self.target = data[test_id]
            self.ct = data[test_id]


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = self.target[idx]
        # ct = self.ct[idx]
        target = normalize_max_min(pad_and_center_image(target,(160,160)))

        # ct = normalize(ct)

        target = torch.Tensor(target)
        # ct = torch.Tensor(ct)
        target = target.unsqueeze(0)
        ct = torch.zeros_like(target)
        # ct = ct.unsqueeze(0)

        return target,target,ct  
    
class MyDatasetRawH(Dataset):
    def __init__(self, is_train):

        self.target = []

        # n_examples = len(data)
        # # n_examples = 1000
        # n_train = int(n_examples * 0.8)  
 

        # 5.1 训练还是测试数据集
        if is_train:
            data = np.load(os.path.join(_HECKTOR_ROOT, 'HECKTOR-train.npy'))
            self.target = data
            self.ct = data


        else:
            data = np.load(os.path.join(_HECKTOR_ROOT, 'HECKTOR-test.npy'))
            self.target = data
            self.ct = data


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = self.target[idx]
        # ct = self.ct[idx]
        target = normalize_max_min(pad_and_center_image(target,(160,160)))

        # ct = normalize(ct)

        target = torch.Tensor(target)
        # ct = torch.Tensor(ct)
        target = target.unsqueeze(0)
        ct = torch.zeros_like(target)
        # ct = ct.unsqueeze(0)

        return target,target,ct  

class MyDatasetRaw(Dataset):
    def __init__(self, is_train):

        self.target = []


        data = np.load('./ctpetraw.npz')

        n_examples = len(data['PET'])
        # n_examples = 1000
        n_train = int(n_examples * 0.8)  
 

        # 5.1 训练还是测试数据集
        if is_train:
            self.target = data['PET'][:n_train]
            self.ct = data['CT'][:n_train]


        else:
            self.target = data['PET'][n_train:]
            self.ct = data['CT'][n_train:]


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = self.target[idx]
        ct = self.ct[idx]
        target = normalize(target)
        ct = normalize(ct)


            

        target = torch.Tensor(target)
        ct = torch.Tensor(ct)
        target = target.unsqueeze(0)
        ct = ct.unsqueeze(0)

        return target,target,ct


from torch_radon_impl import ParallelBeam
def PET2Sinogram(img,radon,count=5e4):
    # count 是计数 用于控制噪声 count高 100e4 就是低噪声 count低20e4 就是低噪声
    if isinstance(img,np.ndarray):
        x = torch.from_numpy(img).float().cuda()
    elif isinstance(img,torch.Tensor):
        x = img.float().cuda()
    else:
        x = torch.tensor(img).float().cuda()
    proj = radon.forward(x)
    mul_factor = torch.ones_like(proj)
    mul_factor = mul_factor+(torch.rand_like(mul_factor)*0.2-0.1)
    noise = torch.ones_like(proj)*torch.mean(mul_factor*proj,dim=(-1,-2),keepdims=True) *0.2
    sinogram = mul_factor*proj + noise
    cs = count/(1e-9+torch.sum(sinogram,dim=(-1,-2),keepdim=True))
    sinogram = sinogram*cs
    mul_factor = mul_factor*cs
    noise = noise*cs
    x = torch.poisson(sinogram)
    return x.reshape(-1,1),mul_factor.reshape(-1,1),noise.reshape(-1,1)
class MyDatasetRawSino(Dataset):
    def __init__(self, is_train):

        self.target = []


        data = np.load('./ctpetraw.npz')

        n_examples = len(data['PET'])
        # n_examples = 1000
        n_train = int(n_examples * 0.8)  
 

        # 5.1 训练还是测试数据集
        if is_train:
            self.target = data['PET'][:n_train]

        else:
            self.target = data['PET'][n_train:]
        # torch.manual_seed(42)



    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = self.target[idx]

        pet_gt1 = np.clip(target/30000,0,1)


        # 投影矩阵
        radon = ParallelBeam(160,np.linspace(0,np.pi,160,False))
        # 从PET图像中仿真正弦图（sinogram）
        sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=50e4)

        sinogram = torch.Tensor(sinogram).unsqueeze(0)
        mul_factor = torch.Tensor(mul_factor).unsqueeze(0)
        pet_noise = torch.Tensor(pet_noise).unsqueeze(0)
            
        input = torch.cat([sinogram,mul_factor,pet_noise],dim=0)

        target = torch.Tensor(pet_gt1)
        target = target.unsqueeze(0)

        return input.detach().cpu(),target
    


class MyDatasetRawBoxBrain(Dataset):
    def __init__(self, is_train,dose=20):

        self.target = []


        # data = np.load('./ctpetraw.npz')
        data = np.load('D:/MIAProject/brainweb/pet_vol_slice_144by144_nt.npy')


        train_id = np.load('./train_list1.npy')
        test_id = np.load('./test_list1.npy')


        # 5.1 训练还是测试数据集
        if is_train:
            self.target = data[train_id]
            train_data = np.load('./newDataset/osem-'+str(dose)+'e4-30-1-train.npz')['osem']
            train_box = np.load('./newDataset/osem-'+str(dose)+'e4-30-1-box-train.npz')['boxes']
            self.input = train_data
            self.boxes = train_box

        else:
            self.target = data[test_id]
            test_data = np.load('./newDataset/osem-'+str(dose)+'e4-30-1-test.npz')['osem']
            test_box = np.load('./newDataset/osem-'+str(dose)+'e4-30-1-box-test.npz')['boxes']
            self.input = test_data
            self.boxes = test_box
        # torch.manual_seed(42)


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = normalize(self.target[idx])

        target = pad_and_center_image(target,(128,128))

        # pet_gt1 = normalize(target)
        # pet_gt1 = np.clip(target/30000,0,1)
        pet_gt1 = target
            
        input = self.input[idx]

        target = torch.Tensor(pet_gt1)
        target = target.unsqueeze(0)

        input = torch.Tensor(input).reshape((1,128,128))
        box = torch.Tensor(self.boxes[idx]/1000*256)

        return input,box, target
    
class MyDatasetHRawBoxBrain(Dataset):
    def __init__(self, is_train,dose=20):

        self.target = []


        # data = np.load('./ctpetraw.npz')
        # data = np.load('D:/MIAProject/brainweb/pet_vol_slice_144by144_nt.npy')


        # train_id = np.load('./train_list1.npy')
        # test_id = np.load('./test_list1.npy')


        # 5.1 训练还是测试数据集
        if is_train:
            data = np.load(os.path.join(_HECKTOR_ROOT, 'HECKTOR-train.npy'))
            self.target = data
            train_data = np.load('./newDataset/osem-h-'+str(dose)+'e4-30-1-train.npz')['osem']
            train_box = np.load('./newDataset/osem-h-'+str(dose)+'e4-30-1-box-train.npz')['boxes']
            self.input = train_data
            self.boxes = train_box

        else:
            data = np.load(os.path.join(_HECKTOR_ROOT, 'HECKTOR-test.npy'))
            self.target = data
            test_data = np.load('./newDataset/osem-h-'+str(dose)+'e4-30-1-test.npz')['osem']
            test_box = np.load('./newDataset/osem-h-'+str(dose)+'e4-30-1-box-test.npz')['boxes']
            self.input = test_data
            self.boxes = test_box
        # torch.manual_seed(42)


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = normalize(self.target[idx])

        target = pad_and_center_image(target,(128,128))

        # pet_gt1 = normalize(target)
        # pet_gt1 = np.clip(target/30000,0,1)
        pet_gt1 = target
            
        input = self.input[idx]

        target = torch.Tensor(pet_gt1)
        target = target.unsqueeze(0)

        input = torch.Tensor(input).reshape((1,128,128))
        box = torch.Tensor(self.boxes[idx]/1000*256)

        return input,box, target



class MyDatasetRawSinoBrain(Dataset):
    def __init__(self, is_train):

        self.target = []


        # data = np.load('./ctpetraw.npz')
        data = np.load('D:/MIAProject/brainweb/pet_vol_slice_144by144_nt.npy')


        train_id = np.load('./train_list1.npy')
        test_id = np.load('./test_list1.npy')

        # 5.1 训练还是测试数据集
        if is_train:
            self.target = data[train_id]

        else:
            self.target = data[test_id ]
        # torch.manual_seed(42)



    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = normalize(self.target[idx])

        target = pad_and_center_image(target,(128,128))

        # pet_gt1 = normalize(target)
        # pet_gt1 = np.clip(target/30000,0,1)
        pet_gt1 = target


        # 投影矩阵
        radon = ParallelBeam(128,np.linspace(0,np.pi,128,False))
        # 从PET图像中仿真正弦图（sinogram）
        sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=50e4)

        sinogram = torch.Tensor(sinogram).unsqueeze(0)
        mul_factor = torch.Tensor(mul_factor).unsqueeze(0)
        pet_noise = torch.Tensor(pet_noise).unsqueeze(0)
            
        input = torch.cat([sinogram,mul_factor,pet_noise],dim=0)

        target = torch.Tensor(pet_gt1)
        target = target.unsqueeze(0)

        return input.detach().cpu(),target

class MyDatasetHRawSinoBrain(Dataset):
    def __init__(self, is_train):

        self.target = []


        # data = np.load('./ctpetraw.npz')

        # 5.1 训练还是测试数据集
        if is_train:
            data = np.load(os.path.join(_HECKTOR_ROOT, 'HECKTOR-train.npy'))
            self.target = data

        else:
            data = np.load(os.path.join(_HECKTOR_ROOT, 'HECKTOR-test.npy'))
            self.target = data
        # torch.manual_seed(42)



    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        target = normalize(self.target[idx])

        target = pad_and_center_image(target,(128,128))

        # pet_gt1 = normalize(target)
        # pet_gt1 = np.clip(target/30000,0,1)
        pet_gt1 = target


        # 投影矩阵
        radon = ParallelBeam(128,np.linspace(0,np.pi,128,False))
        # 从PET图像中仿真正弦图（sinogram）
        sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=20e4)

        sinogram = torch.Tensor(sinogram).unsqueeze(0)
        mul_factor = torch.Tensor(mul_factor).unsqueeze(0)
        pet_noise = torch.Tensor(pet_noise).unsqueeze(0)
            
        input = torch.cat([sinogram,mul_factor,pet_noise],dim=0)

        target = torch.Tensor(pet_gt1)
        target = target.unsqueeze(0)

        return input.detach().cpu(),target


class PETCTDataset(Dataset):
    def __init__(self, is_train, use_ct = False):

        if is_train:
            data = np.load('./dataSetK3-CT-KEM-192-Train.npz')
            self.input = data['input']
            self.target = data['target']


        else:
            data = np.load('./dataSetK3-CT-KEM-192-Test.npz')
            self.input = data['input']
            self.target = data['target']

        if use_ct == False:
            self.input = self.input[:,1:4,:,:]


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        input = self.input[idx]
        target = self.target[idx]

        # print(np.max(input),np.min(input),np.max(target),np.min(target))

        input = torch.tensor(input).float()
        target = torch.tensor(target).float()

        return input,target    
    
def normalize_max_min(img):
    Max = np.max(img)
    Min = np.min(img)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化
class MouseDataset(Dataset):
    def __init__(self, is_train, use_ct = False):


        data = np.load('./mouseDatset.npz')
        self.input = data['input']
        self.target = data['target']
        
        self.input = np.swapaxes(data['input'][35:],0,-1)
        self.target = np.swapaxes(data['target'][35:],0,-1)



        print(self.input.shape)


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        input = normalize_max_min(self.input[idx])
        # target = normalize_max_min(self.target[idx])
        input = self.input[idx]
        target = self.target[idx]

        input = torch.tensor(input).float()
        target = torch.tensor(target).float()

        return input,target    
    

class MouseDatasetRaw(Dataset):
    def __init__(self, is_train, use_ct = False):


        data = np.load('./mouseDatset.npz')

        self.input = data['input']
        self.target = data['target']

        # self.input = np.swapaxes(data['input'][35:],0,-1)
        # self.target = np.swapaxes(data['target'][35:],0,-1)




        print(self.input.shape)


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        # input = normalize_max_min(self.input[idx])
        # target = normalize_max_min(self.target[idx])
        target = self.target[idx]
        target = pad_and_center_image(target,(160,160))
        # target = np.clip(target* 100,0,1)
        target = normalize_max_min(self.target[idx])

        # target = normalize(target)



            

        target = torch.Tensor(target)

        target = target.unsqueeze(0)


        return target,target
    




class HECKTORdataset(Dataset):
    def __init__(self,  mode='train', transform=None,data_rate = 1.0,enhence_rate = 0):
        self.transform = transform  # using transform in torch!
        self.mode = mode
        self.inputs = []
        self.targets = []
        data = np.load(os.path.join(_HECKTOR_ROOT, 'HECKTOR.npz'))

        np.random.seed(42)  
        random.seed(42)
        n_examples = len(data['input'])
        # n_examples = 1000
        n_train = n_examples * 0.8  
        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        test_idx = list(set(range(0,n_examples))-set(train_idx))
        n = int(np.round(len(train_idx)) * data_rate)
        train_idx = train_idx[:n]

        print(n_examples,len(train_idx),len(test_idx))
        if self.mode == 'train':
            inputs_cases = data['input'][train_idx]
            targets_cases = data['target'][train_idx]
            # inputs_cases = data['input'][0:180]
            # targets_cases = data['target'][0:180]
            # print(inputs_cases.shape,targets_cases.shape)
            # plt.imshow(inputs_cases[0,0,:,:,0])
            # plt.show()
            # plt.imshow(inputs_cases[0,:,:,0,0])
            # plt.show()
            qbar = trange(len(inputs_cases))
            for i in range(len(inputs_cases)):
                for j in range(len(inputs_cases[i])):
                    tz = np.sum(targets_cases[i,:,:,j])
                    if tz > 0:
                        #original
                        input = inputs_cases[i,:,:,j,:]
                        target = targets_cases[i,:,:,j]
                        self.inputs.append(input.copy())
                        self.targets.append(target.copy())
                        # input[:, :, 0] = np.clip(standerlize(input[:, :, 0]),0,1)  # ct
                        # input[:, :, 1] = np.clip(standerlize(input[:, :, 1]),0,1)  # pt
                        #
                        # inputS,targetS = upsample(input,target)
                        # self.inputs.append(inputS.copy())
                        # self.targets.append(targetS.copy())
                qbar.update(1)
        else:
            inputs_cases = data['input'][test_idx]
            targets_cases = data['target'][test_idx]
            # inputs_cases = data['input'][180:]
            # targets_cases = data['target'][180:]
            qbar1 = trange(len(inputs_cases))
            for i in range(len(inputs_cases)):
                for j in range(len(inputs_cases[i])):
                    if np.sum(targets_cases[i,:,:,j]) > 0:
                        input = inputs_cases[i,:,:,j,:]
                        target = targets_cases[i,:,:,j]

                        self.inputs.append(input )
                        self.targets.append(target)
                        # inputS,targetS = upsample(input,target)
                        # self.inputs.append(inputS.copy())
                        # self.targets.append(targetS.copy())
                qbar1.update(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = dict()

        sample['id'] = idx  # hecktor

        # label = np.expand_dims(self.targets[idx], axis=2)
        sample['input'] = self.inputs[idx].transpose([1, 0, 2])
        # sample['target'] = label.transpose([1, 0, 2])

        target = torch.Tensor(pad_and_center_image(self.inputs[idx].transpose([1, 0, 2])[1],(160,160))).unsqueeze(0)
        ct = torch.zeros_like(target)
        # if self.transform:
        #     sample = self.transform(sample)
        return target,target,ct