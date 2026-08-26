import numpy as np
import scipy.sparse
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
def normalize(I):
    if isinstance(I,np.ndarray):
        normI = (I-np.amin(I,axis=(-1,-2)))/(np.amax(I,axis=(-1,-2))-np.amin(I,axis=(-1,-2)))
    elif isinstance(I,torch.Tensor):
        normI = (I-torch.amin(I,dim=(-1,-2)))/(torch.amax(I,dim=(-1,-2))-torch.amin(I,dim=(-1,-2)))
    else:
        raise TypeError("I must be cupy.ndarray or numpy.ndarray or torch.Tensor")
    return normI
def proj_forward(img,radon=None,G=None):
    assert radon is not None or G is not None,"proj_forward, radon or G must not be None"
    if radon is not None:
        assert np.prod(img.shape)%(radon.volume.height*radon.volume.width)==0,f"img should can reshape to height {radon.volume.height}, width {radon.volume.width},get shape {img.shape}"
        if isinstance(img,np.ndarray):
            out = radon.forward(torch.from_numpy(img).reshape(np.prod(img.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        elif isinstance(img,torch.Tensor):
            out = radon.forward(img.reshape(np.prod(img.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        return out.reshape(np.prod(img.shape)//(radon.volume.height*radon.volume.width),-1).T
    elif G is not None:
        assert np.prod(img.shape)%G.shape[1]==0,f"img should can reshape to height {G.shape[1]},get shape {img.shape}"
        if isinstance(img,np.ndarray):
            out = G.dot(img.reshape(-1,np.prod(img.shape)//G.shape[1]))
        elif isinstance(img,torch.Tensor):
            out = G.dot(img.reshape(-1,np.prod(img.shape)//G.shape[1]).cpu().numpy())
        return out
    else:
        raise ValueError("proj_forward, radon or G must not be None")
def proj_backward(sinogram,radon=None,G=None):
    assert radon is not None or G is not None,"proj_backward, radon or G must not be None"
    if radon is not None:
        assert np.prod(sinogram.shape)%(radon.volume.height*radon.volume.width)==0,f"sinogram should can reshape to height {radon.volume.height}, width {radon.volume.width},get shape {sinogram.shape}"
        if isinstance(sinogram,np.ndarray):
            out = radon.backward(torch.from_numpy(sinogram).reshape(np.prod(sinogram.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        elif isinstance(sinogram,torch.Tensor):
            out = radon.backward(sinogram.reshape(np.prod(sinogram.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        return out.reshape(np.prod(sinogram.shape)//(radon.volume.height*radon.volume.width),-1).T
    elif G is not None:
        assert np.prod(sinogram.shape)%G.shape[0]==0,f"sinogram should can reshape to height {G.shape[0]},get shape {sinogram.shape}"
        if isinstance(sinogram,np.ndarray):
            out = G.T.dot(sinogram.reshape(-1,np.prod(sinogram.shape)//G.shape[0]))
        elif isinstance(sinogram,torch.Tensor):
            out = G.T.dot(sinogram.reshape(-1,np.prod(sinogram.shape)//G.shape[0]).cpu().numpy())
        return out
    else:
        raise ValueError("proj_backward, radon or G must not be None")
def mynp_divide(a,b,out=None,where=None):
    if where is not None and out is not None:
        return np.where(where,np.divide(a,b),out)
    elif where is None:
        return np.divide(a,b,out)
    else:
        out = np.zeros_like(a)
        return np.where(where,np.divide(a,b),out)
def one_iter(sinogram,mul_factor,pet_noise,G,K,G_row_SumofCol,pet_eval,pet_gt,method_index='KEM',radon=None):
    # print(sinogram.shape)
    if K is None:
        K = scipy.sparse.identity(np.prod(pet_gt.shape), format='coo')
    tmp_pet_eval = K.dot(pet_eval)
    if radon is not None:
        l = proj_forward(tmp_pet_eval.reshape(pet_gt.shape),radon)
    else:
        l = G.dot(tmp_pet_eval)
    sinogram_eval = mul_factor*l + pet_noise
    sinogram_mean = torch.mean(sinogram) if isinstance(sinogram, torch.Tensor) else np.mean(sinogram)
    tmp = mynp_divide(sinogram,sinogram_eval+sinogram_mean*1e-9,out=np.ones_like(sinogram_eval))
    tmp[np.logical_and(sinogram==0,sinogram_eval==0)]=1
    if method_index == 'KEM':
        if radon is not None:
            backproj = K.T.dot(proj_backward(mul_factor*tmp,radon))
            pet_eval = pet_eval/(K.T.dot(G_row_SumofCol) + 1e-5)*backproj
        else:
            backproj = K.T.dot(G.T.dot(mul_factor*tmp))
            pet_eval = pet_eval/(K.T.dot(G_row_SumofCol))*backproj
    elif method_index == 'MLEM':
        if radon is not None:
            backproj = proj_backward(mul_factor*tmp,radon)
            pet_eval = pet_eval/(G_row_SumofCol)*backproj
        else:
            backproj = G.T.dot(mul_factor*tmp)
            pet_eval = pet_eval/(G_row_SumofCol)*backproj
    else:
        raise ValueError("method_index must be 'KEM' or 'MLEM'")
    pet_eval[G_row_SumofCol==0] = 0
    return pet_eval

def MLEMorKEM(sinogram,mul_factor,pet_noise,G,pet_gt,K=None,method_index='KEM',prefix='',radon=None,init=None,total_iter=None):
    if init is None:
        pet_eval = np.ones_like(pet_gt).reshape(-1,1)
    else:
        pet_eval = init.reshape(-1,1)
    if radon is not None:
        G_row_SumofCol = proj_backward(mul_factor,radon)
    else:
        G_row_SumofCol = G.T.dot(mul_factor)

    if total_iter is None:
        if K is not None:
            total_iter=60
        else:
            total_iter = 25
    if K is None:
        K =scipy.sparse.identity(np.prod(pet_gt.shape), format='coo')

    imgs = [K.dot(pet_eval).reshape(pet_gt.shape)]

    for iter in range(total_iter):
        pet_eval = one_iter(sinogram,mul_factor,pet_noise,G,K,G_row_SumofCol,pet_eval,pet_gt,method_index,radon)
        img = K.dot(pet_eval).reshape(pet_gt.shape)
        # img = normalize(img)*np.max(pet_gt)
        # img = img.clip(0,np.max(pet_gt))
        imgs.append(img)


        img = (normalize(img)*255).astype(np.uint8)
    if total_iter <= 10:
        return imgs[-1]
    elif total_iter <= 20:
        return imgs[-1], imgs[-11]
    elif total_iter <= 30:
        return imgs[-1], imgs[-11], imgs[-21]
    elif total_iter <= 40:
        return imgs[-1], imgs[-11], imgs[-21], imgs[-31]

    return imgs[-1],imgs[-11],imgs[-21],imgs[-31],imgs[-41]

def returnResult_MLEMorKEM(sinogram,mul_factor,pet_noise,G,pet_gt,K=None,method_index='KEM',prefix='',radon=None,init=None,total_iter=None):
    if init is None:
        pet_eval = np.ones_like(pet_gt).reshape(-1,1)
    else:
        pet_eval = init.reshape(-1,1)
    if radon is not None:
        G_row_SumofCol = proj_backward(mul_factor,radon)
    else:
        G_row_SumofCol = G.T.dot(mul_factor)
    gt = (normalize(pet_gt)*255).astype(np.uint8)
    if total_iter is None:
        if K is not None:
            total_iter=60
        else:
            total_iter = 25
    if K is None:
        K =scipy.sparse.identity(np.prod(pet_gt.shape), format='coo')

    imgs = [K.dot(pet_eval).reshape(pet_gt.shape)]
    psnrs = [compare_psnr(pet_gt, imgs[0],data_range=np.max(pet_gt))]
    ssims = [compare_ssim(pet_gt, imgs[0],data_range=np.max(pet_gt))]
    mses = [compare_mse(pet_gt, imgs[0])]
    norm_psnrs = [compare_psnr(gt, (normalize(imgs[0])*255).astype(np.uint8))]
    norm_ssims = [compare_ssim(gt, (normalize(imgs[0])*255).astype(np.uint8))]
    norm_mses = [compare_mse(gt, (normalize(imgs[0])*255).astype(np.uint8))]
    for iter in range(total_iter):
        pet_eval = one_iter(sinogram,mul_factor,pet_noise,G,K,G_row_SumofCol,pet_eval,pet_gt,method_index,radon)
        img = K.dot(pet_eval).reshape(pet_gt.shape)
        # img = normalize(img)*np.max(pet_gt)
        # img = img.clip(0,np.max(pet_gt))
        imgs.append(img)
        psnr = compare_psnr(pet_gt, img,data_range=np.max(pet_gt))
        ssim = compare_ssim(pet_gt, img,data_range=np.max(pet_gt))
        mse = compare_mse(pet_gt, img)
        psnrs.append(psnr)
        ssims.append(ssim)
        mses.append(mse)
        img = (normalize(img)*255).astype(np.uint8)
        norm_psnr = compare_psnr(gt, img)
        norm_ssim = compare_ssim(gt, img)
        norm_mse = compare_mse(gt, img)
        norm_psnrs.append(norm_psnr)
        norm_ssims.append(norm_ssim)
        norm_mses.append(norm_mse)
    return np.max(psnrs),np.max(ssims),np.min(mses),imgs[-1]
def read_data(filename):
    import h5py
    data = h5py.File(filename,'r')
    return data
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
def PET2Sinogram(img,radon,count=5e4):
    raise RuntimeError(
        "retired PET simulator: its random gain and flat additive background "
        "are not part of the declared study model; use `sam-kem simulate`"
    )
    # Historical implementation retained below for forensic inspection only.
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
    return x.detach().cpu().numpy().reshape(-1,1),mul_factor.detach().cpu().numpy().reshape(-1,1),noise.detach().cpu().numpy().reshape(-1,1)
import random
from torch_radon_impl import ParallelBeam
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix

def gaussian_kernel2(X,Y,kernel_size, sigma, J):

    x_hat = X.reshape(-1, 1)
    y_hat = Y.reshape(-1, 1)
    rows = []
    cols = []
    data = []

    for i in range(kernel_size):
        # 计算当前位置的邻域
        for j in range(-J, J+1):
            for k in range(-J, J+1):
                row = i // 128 + j
                col = i % 128 + k
                # 确保索引在矩阵范围内
                if 0 <= row < 128 and 0 <= col < 128:
                    dist = np.abs(j) + np.abs(k)
                    if dist <= J:
                        index = row * 128 + col
                        rows.append(i)
                        cols.append(index)
                        data.append(np.exp(-(x_hat[i] - y_hat[index])**2 / (2 * sigma**2)))
    # 确保数组是一维的
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data).reshape(-1)
    # print(rows.shape,cols.shape,data.shape)
    # 创建稀疏矩阵
    kernel = coo_matrix((data, (rows, cols)), shape=(kernel_size, kernel_size))
    return kernel

def bowsher_kernel(mr_image, kernel_size, sigma_mr, J):
    """Build Bowsher prior kernel from MR intensity similarity only.

    Unlike gaussian_kernel which uses PET features, Bowsher computes
    pairwise weights purely from the reference (MR) image intensities.
    This is the classic anatomy-guided penalty from Bowsher et al.

    Args:
        mr_image: 2D reference image (128x128), flattened internally.
        kernel_size: total number of pixels (128*128).
        sigma_mr: intensity similarity bandwidth for MR.
        J: spatial neighborhood radius (Manhattan distance).

    Returns:
        Sparse COO matrix of shape (kernel_size, kernel_size).
    """
    mr_flat = mr_image.reshape(-1)
    rows = []
    cols = []
    data = []

    for i in range(kernel_size):
        for j in range(-J, J+1):
            for k in range(-J, J+1):
                row = i // 128 + j
                col = i % 128 + k
                if 0 <= row < 128 and 0 <= col < 128:
                    dist = abs(j) + abs(k)
                    if dist <= J:
                        index = row * 128 + col
                        rows.append(i)
                        cols.append(index)
                        data.append(np.exp(-(mr_flat[i] - mr_flat[index])**2 / (2 * sigma_mr**2)))

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data).reshape(-1)
    kernel = coo_matrix((data, (rows, cols)), shape=(kernel_size, kernel_size))
    return kernel

def gaussian_kernel(X,kernel_size, sigma, J):

    x_hat = X.reshape(-1, 1)
    rows = []
    cols = []
    data = []

    for i in range(kernel_size):
        # 计算当前位置的邻域
        for j in range(-J, J+1):
            for k in range(-J, J+1):
                row = i // 128 + j
                col = i % 128 + k
                # 确保索引在矩阵范围内
                if 0 <= row < 128 and 0 <= col < 128:
                    dist = np.abs(j) + np.abs(k)
                    if dist <= J:
                        index = row * 128 + col
                        rows.append(i)
                        cols.append(index)
                        data.append(np.exp(-(x_hat[i] - x_hat[index])**2 / (2 * sigma**2)))
    # 确保数组是一维的
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data).reshape(-1)
    # print(rows.shape,cols.shape,data.shape)
    # 创建稀疏矩阵
    kernel = coo_matrix((data, (rows, cols)), shape=(kernel_size, kernel_size))
    return kernel

    X = torch.Tensor(X).cuda()
    # 将X展平为一维张量，并移至GPU
    x_hat = X.view(-1).cuda()
    
    # 计算所有点对的距离矩阵
    rows = torch.arange(kernel_size, dtype=torch.long, device='cuda')
    cols = torch.arange(kernel_size, dtype=torch.long, device='cuda')
    rows_grid, cols_grid = torch.meshgrid(rows, cols)
    dist_matrix = (torch.abs(rows_grid - cols_grid) % 128 + torch.abs((rows_grid - cols_grid) // 128)).to('cuda')
    
    # 创建一个与距离矩阵同样大小的布尔张量，标记距离小于J的位置
    mask = dist_matrix < J
    
    # 只计算距离小于J的元素对应的核矩阵值
    x_hat_diff = x_hat[rows_grid[mask]] - x_hat[cols_grid[mask]]
    kernel_values = torch.exp(-(x_hat_diff ** 2) / (2 * sigma ** 2))
    
    # 初始化核矩阵
    kernel = torch.zeros((kernel_size, kernel_size), device='cuda')
    
    # 将计算得到的核矩阵值赋给对应的位置
    kernel[mask] = kernel_values
    
    return kernel

    # 将X展平为一维数组
    x_hat = X.flatten()
    
    # 计算所有点对的距离矩阵
    rows = np.arange(kernel_size)
    cols = np.arange(kernel_size)
    rows_grid, cols_grid = np.meshgrid(rows, cols)
    dist_matrix = np.abs(rows_grid - cols_grid) % 128 + np.abs((rows_grid - cols_grid) // 128)
    
    # 创建一个与距离矩阵同样大小的布尔矩阵，标记距离小于J的位置
    mask = dist_matrix < J
    
    # 只计算距离小于J的元素对应的核矩阵值
    x_hat_diff = x_hat[rows_grid[mask]] - x_hat[cols_grid[mask]]
    kernel_values = np.exp(-(x_hat_diff ** 2) / (2 * sigma ** 2))
    
    # 初始化核矩阵
    kernel = np.zeros((kernel_size, kernel_size))
    
    # 将计算得到的核矩阵值赋给对应的位置
    kernel[mask] = kernel_values

    # 由于核矩阵是对称的，所以将上三角复制到下三角
    kernel = kernel + kernel.T - np.diag(np.diag(kernel))
    
    return kernel

    x_hat = X.reshape(-1,1)
    kernel = np.zeros((kernel_size,kernel_size))
    for i in range(kernel_size):
        # for j in range(kernel_size):
        #     dist = np.abs(i%128-j%128) + np.abs(i/128 - j/128)
        #     if (dist < J):
        #         kernel[i][j] = np.exp(-(x_hat[i] - x_hat[j])**2 / 2*sigma**2)
        if (i+1 < kernel_size):
            kernel[i][i+1] = np.exp(-(x_hat[i] - x_hat[i+1])**2 / 2*sigma**2)
        if (i+128 < kernel_size):
            kernel[i+1][i] = np.exp(-(x_hat[i] - x_hat[i+128])**2 / 2*sigma**2)
        if (i-1 >= 0):
            kernel[i][i-1] = np.exp(-(x_hat[i] - x_hat[i-1])**2 / 2*sigma**2)
        if (i-128 >= 0):
            kernel[i-1][i] = np.exp(-(x_hat[i] - x_hat[i-128])**2 / 2*sigma**2)

        if (i+128+1 < kernel_size and i+1 < kernel_size):
            kernel[i+1][i+1] = np.exp(-(x_hat[i] - x_hat[i+128+1])**2 / 2*sigma**2)
        if (i+128-1 < kernel_size and i-1 >= 0):
            kernel[i+1][i-1] = np.exp(-(x_hat[i] - x_hat[i+128-1])**2 / 2*sigma**2)
        if (i-128 >= 0 and i+1 < kernel_size):
            kernel[i-1][i+1] = np.exp(-(x_hat[i] - x_hat[i-128+1])**2 / 2*sigma**2)
        if (i-128-1 >= 0 and i-1 >= 0):
            kernel[i-1][i-1] = np.exp(-(x_hat[i] - x_hat[i-128-1])**2 / 2*sigma**2)

    return kernel

    # x_hat = X.reshape(-1,1)
    # # 计算距离矩阵
    # rows, cols = np.indices((kernel_size, kernel_size))
    # dist_matrix = np.abs(rows % 128 - cols % 128) + np.abs(rows // 128 - cols // 128)

    # # 初始化kernel矩阵
    # kernel = np.zeros((kernel_size, kernel_size))

    # # 计算高斯核函数的值
    # gaussian_matrix = np.exp(-(x_hat - x_hat.T)**2 / (2 * sigma**2))

    # # 根据距离矩阵和条件设置kernel的值
    # kernel[dist_matrix < J] = gaussian_matrix[dist_matrix < J]

    # x_hat = X.view(-1, 1)

    # # 计算距离矩阵
    # rows, cols = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    # dist_matrix = torch.abs(rows % 128 - cols % 128) + torch.abs(rows // 128 - cols // 128)

    # # 初始化kernel矩阵
    # kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)

    # # 计算高斯核函数的值
    # gaussian_matrix = torch.exp(-(x_hat - x_hat.t())**2 / (2 * sigma**2))

    # # 根据距离矩阵和条件设置kernel的值
    # kernel[dist_matrix < J] = gaussian_matrix[dist_matrix < J]


    # # 确保 X 是一维的
    # x_hat = X.reshape(-1, 1)
    
    # # 使用 scipy 的 cdist 函数计算距离矩阵
    # dist_matrix = cdist(x_hat, x_hat, metric='cityblock')
    
    # # 创建一个布尔索引，用于选择小于 J 的距离
    # bool_index = dist_matrix < J
    
    # # 在选定的距离上计算高斯核函数的值
    # data = np.exp(-(dist_matrix[bool_index])**2 / (2 * sigma**2))
    
    # # 获取非零元素的行和列索引
    # row, col = np.nonzero(bool_index)
    
    # # 使用非零元素的位置和值创建稀疏矩阵
    # kernel = coo_matrix((data, (row, col)), shape=(kernel_size, kernel_size))

    return kernel


if __name__=="__main__":
    # 读取PET数据
    data1 = read_data("pet_new.mat")
    pet_gt,mr = np.array(data1["pet_image"]).T,np.array(data1["mri_image"]).T
    pet_gt,mr = pad_and_center_image(np.stack([pet_gt,mr], axis=0),(128,128))
    pet_gt1 = pet_gt
    pet_gt1 = normalize(pet_gt1)
    # 投影矩阵
    radon = ParallelBeam(128,np.linspace(0,180,128,False))
    # 从PET图像中仿真正弦图（sinogram）
    sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=200e4)
    radon2 = ParallelBeam(128,np.linspace(0,180,128,False))
    # 使用EM生成PET图像
    sinogram = torch.tensor(sinogram).float()
    pet_gt1 = torch.tensor(pet_gt1).float()
    mul_factor = torch.tensor(mul_factor).float()
    pet_noise = torch.tensor(pet_noise).float()

    kernel_size = 128*128
    sigma = 100 #0.05
    gaussian_matrix = gaussian_kernel2(pet_gt,mr,kernel_size, sigma,J=2)

    # kernel = 1.0 * RBF(1.0)
    # gaussian_matrix = gaussian_matrix.reshape(-1,1)
    print(gaussian_matrix)

    img = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon2,pet_gt=pet_gt1,K=gaussian_matrix,method_index='KEM',prefix='KEM',total_iter=60)
    img2 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon2,pet_gt=pet_gt1,K=None,method_index='KEM',prefix='KEM',total_iter=30)
    # print(psnrs,ssims,mses)
    # 画图
    plt.figure()
    plt.subplot(1,5,1)
    plt.title("sinogram")
    plt.imshow(sinogram.reshape(128,128),cmap="gray")
    plt.subplot(1,5,2)
    plt.title("gaussian kernel matrix")
    plt.imshow(img[0],cmap="gray")
    plt.subplot(1,5,3)
    plt.title("I kernel matrix(MLEM)")
    plt.imshow(img2[0],cmap="gray")
    plt.subplot(1,5,4)
    plt.title("GT")
    plt.imshow(pet_gt,cmap="gray")
    plt.subplot(1,5,5)
    plt.title("GT")
    plt.imshow(mr,cmap="gray")
    plt.show()
