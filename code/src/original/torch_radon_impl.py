"""
Pure PyTorch/numpy implementation of parallel-beam Radon transform.
API-compatible with torch_radon.ParallelBeam, but does NOT depend on torch_radon.

This implementation uses F.affine_grid + F.grid_sample for differentiable
forward and backward projections.

Forward projection: rotate image by -theta, sum along vertical axis -> detector readings.
Back projection: expand sinogram to 2D, rotate by +theta, accumulate over all angles.

Both operations are fully differentiable via F.grid_sample.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


class Volume:
    """Simple volume descriptor matching torch_radon API."""
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width


class ParallelBeam:
    """
    Pure PyTorch parallel beam Radon transform.
    API-compatible with torch_radon.ParallelBeam.

    Usage:
        radon = ParallelBeam(img_size, angles)
        sinogram = radon.forward(images)    # (B, H, W) -> (B, N_angles, N_detectors)
        backproj = radon.backward(sinogram) # (B, N_angles, N_detectors) -> (B, H, W)

    Properties:
        radon.volume.height   - image height
        radon.volume.width    - image width
        radon.det_count       - number of detector bins (= volume_size)
        radon.n_angles        - number of projection angles
    """

    def __init__(self, volume_size: int, angles: Union[np.ndarray, torch.Tensor]):
        """
        Args:
            volume_size: int, the image size (assumes square HxH)
            angles: numpy array or torch tensor of angles in radians
        """
        self.volume_size = volume_size
        self.volume = Volume(volume_size, volume_size)
        self.det_count = volume_size

        if isinstance(angles, np.ndarray):
            self.angles = torch.from_numpy(angles).float()
        else:
            self.angles = angles.float()

        self.n_angles = len(self.angles)

        # Precompute affine grids for all angles on CPU (lazy move to device at runtime)
        # This avoids recreating grids every forward/backward call
        self._grids_forward = None   # grids for forward projection (rotate by -theta)
        self._grids_backward = None  # grids for back projection (rotate by +theta)

    def _get_grid(self, angle: torch.Tensor, B: int, H: int, W: int,
                  dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Build a sampling grid for rotating an image by the given angle.

        Args:
            angle: scalar torch tensor, rotation angle in radians
            B: batch size
            H: image height
            W: image width
            dtype: data type
            device: target device

        Returns:
            grid of shape (B, H, W, 2) for F.grid_sample
        """
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        # 2x3 affine transformation matrix for pure rotation
        theta = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0]
        ], dtype=dtype, device=device)
        grid = F.affine_grid(theta.unsqueeze(0), (B, 1, H, W), align_corners=False)
        return grid

    def _ensure_grids(self, B: int, dtype: torch.dtype, device: torch.device):
        """
        Lazily precompute all grids for forward and backward projections.
        Grids are cached with batch_size=1 and expanded to match the actual B at runtime.
        """
        H = W = self.volume_size

        if self._grids_forward is None or self._grids_forward[0].device != device:
            self._grids_forward = []
            self._grids_backward = []
            for i in range(self.n_angles):
                angle_fwd = torch.tensor(-self.angles[i].item(), dtype=dtype, device=device)
                grid_fwd = self._get_grid(angle_fwd, 1, H, W, dtype, device)
                self._grids_forward.append(grid_fwd)

                angle_bwd = torch.tensor(self.angles[i].item(), dtype=dtype, device=device)
                grid_bwd = self._get_grid(angle_bwd, 1, H, W, dtype, device)
                self._grids_backward.append(grid_bwd)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward projection (image -> sinogram).

        For each angle theta:
            1. Rotate image by -theta
            2. Sum along vertical axis -> gives detector readings (line integrals)

        Args:
            images: torch tensor of shape (B, H, W) or (B, 1, H, W)

        Returns:
            torch tensor of shape (B, N_angles, N_detectors)
        """
        if images.dim() == 2:
            images = images.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif images.dim() == 3:
            images = images.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        B, C, H, W = images.shape
        dtype = images.dtype
        device = images.device

        self._ensure_grids(B, dtype, device)

        sinograms = []
        for i in range(self.n_angles):
            # Expand grid from (1, H, W, 2) to (B, H, W, 2)
            grid_i = self._grids_forward[i].expand(B, -1, -1, -1)
            # Rotate image by -angle_i
            rotated = F.grid_sample(
                images, grid_i,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            # Sum along vertical axis (dim=2) -> (B, C, W)
            detector = rotated.sum(dim=2)
            sinograms.append(detector)

        sinogram = torch.cat(sinograms, dim=1)  # (B, N_angles, W)
        return sinogram

    def backward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Back projection (sinogram -> image).

        For each angle theta:
            1. Expand sinogram column to 2D (replicate along y-axis)
            2. Rotate by +theta
            3. Accumulate into the image

        This is the adjoint of the forward projection.

        Args:
            sinogram: torch tensor of shape (B, N_angles, N_detectors).
                Also accepts extra singleton dimensions, e.g. (B, 1, N_angles, N_detectors)
                or (B, 1, N_angles, N_detectors, 1) — these are squeezed automatically.

        Returns:
            torch tensor of shape (B, H, W)
        """
        # Handle extra singleton dimensions (e.g. from broadcasting in ADMMNet)
        while sinogram.dim() > 3:
            # Try to squeeze singleton dimensions
            squeezed = False
            for d in range(sinogram.dim()):
                if sinogram.shape[d] == 1:
                    sinogram = sinogram.squeeze(d)
                    squeezed = True
                    break
            if not squeezed:
                # No more singleton dims but still > 3D: flatten all but last 2 dims into B
                B_new = 1
                for d in range(sinogram.dim() - 2):
                    B_new *= sinogram.shape[d]
                sinogram = sinogram.reshape(B_new, sinogram.shape[-2], sinogram.shape[-1])
                break

        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)

        B = sinogram.shape[0]
        H = W = self.volume_size
        dtype = sinogram.dtype
        device = sinogram.device

        self._ensure_grids(B, dtype, device)

        # Initialize back projection
        bp = torch.zeros(B, 1, H, W, dtype=dtype, device=device)

        for i in range(self.n_angles):
            # Get sinogram for this angle: (B, 1, N_det) -> (B, 1, 1, N_det)
            sino_i = sinogram[:, i:i + 1].unsqueeze(2)
            # Expand to 2D: replicate along image height -> (B, 1, H, W)
            sino_2d = sino_i.expand(-1, -1, H, -1)
            # Expand grid from (1, H, W, 2) to (B, H, W, 2)
            grid_i = self._grids_backward[i].expand(B, -1, -1, -1)
            # Rotate back by +angle_i
            rotated = F.grid_sample(
                sino_2d, grid_i,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            bp = bp + rotated

        return bp.squeeze(1)  # (B, H, W)


# ============================================================
# Test / verification code
# ============================================================
if __name__ == '__main__':
    import time

    print("=" * 60)
    print("torch_radon_impl.py - 纯PyTorch Radon变换实现测试")
    print("=" * 60)

    # ---- 基本参数 ----
    img_size = 128
    n_angles = 128
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    # ---- 创建Radon变换对象 ----
    radon = ParallelBeam(img_size, angles)
    print(f"\n[1] API检查:")
    print(f"    volume.height = {radon.volume.height}")
    print(f"    volume.width  = {radon.volume.width}")
    print(f"    det_count     = {radon.det_count}")
    print(f"    n_angles      = {radon.n_angles}")

    # ---- 生成测试图像 (phantom) ----
    # 创建一个简单的圆盘phantom
    y, x = np.ogrid[-1:1:img_size * 1j, -1:1:img_size * 1j]
    phantom_np = (x ** 2 + y ** 2 < 0.5 ** 2).astype(np.float32)
    # 添加内部结构
    phantom_np += 0.5 * ((x ** 2 + y ** 2 < 0.3 ** 2).astype(np.float32))
    phantom = torch.from_numpy(phantom_np).unsqueeze(0)  # (1, H, W)

    print(f"\n[2] 前向投影测试:")
    print(f"    输入图像形状: {phantom.shape}")

    t0 = time.time()
    sinogram = radon.forward(phantom)
    t_fwd = time.time() - t0
    print(f"    输出sinogram形状: {sinogram.shape}")
    print(f"    期望形状: (1, {n_angles}, {img_size})")
    assert sinogram.shape == (1, n_angles, img_size), "前向投影形状错误!"
    print(f"    前向投影耗时: {t_fwd:.3f}s")

    print(f"\n[3] 反向投影测试:")
    print(f"    输入sinogram形状: {sinogram.shape}")

    t0 = time.time()
    backproj = radon.backward(sinogram)
    t_bwd = time.time() - t0
    print(f"    输出图像形状: {backproj.shape}")
    print(f"    期望形状: (1, {img_size}, {img_size})")
    assert backproj.shape == (1, img_size, img_size), "反向投影形状错误!"
    print(f"    反向投影耗时: {t_bwd:.3f}s")

    print(f"\n[4] 可微性测试 (梯度传播):")
    phantom_grad = phantom.clone().requires_grad_(True)
    sino = radon.forward(phantom_grad)
    loss = sino.sum()
    loss.backward()
    has_grad = phantom_grad.grad is not None
    print(f"    梯度存在: {has_grad}")
    assert has_grad, "梯度传播失败! F.grid_sample不支持自动微分."
    print(f"    梯度形状: {phantom_grad.grad.shape}")
    print(f"    梯度非零元素数: {(phantom_grad.grad != 0).sum().item()}")

    print(f"\n[5] 批量输入测试:")
    batch_phantom = phantom.expand(4, -1, -1)  # (4, H, W)
    sino_batch = radon.forward(batch_phantom)
    bp_batch = radon.backward(sino_batch)
    print(f"    批量前向: 输入 {batch_phantom.shape} -> 输出 {sino_batch.shape}")
    assert sino_batch.shape == (4, n_angles, img_size), "批量前向形状错误!"
    print(f"    批量反向: 输入 {sino_batch.shape} -> 输出 {bp_batch.shape}")
    assert bp_batch.shape == (4, img_size, img_size), "批量反向形状错误!"

    print(f"\n[6] 前向-反向一致性检查:")
    # 对于均匀的圆盘, backproj(forward(image)) 应该是一个模糊版本
    bp_norm = torch.norm(backproj).item()
    phantom_norm = torch.norm(phantom).item()
    print(f"    原始图像范数: {phantom_norm:.4f}")
    print(f"    前向-反向后范数: {bp_norm:.4f}")
    print(f"    比值 (backproj/forward): {bp_norm / phantom_norm:.4f}")

    print(f"\n[7] GPU加速测试 (如果可用):")
    if torch.cuda.is_available():
        phantom_gpu = phantom.cuda()
        radon_gpu = ParallelBeam(img_size, angles)

        torch.cuda.synchronize()
        t0 = time.time()
        sinogram_gpu = radon_gpu.forward(phantom_gpu)
        torch.cuda.synchronize()
        t_fwd_gpu = time.time() - t0
        print(f"    GPU前向投影耗时: {t_fwd_gpu:.3f}s (CPU: {t_fwd:.3f}s)")

        torch.cuda.synchronize()
        t0 = time.time()
        backproj_gpu = radon_gpu.backward(sinogram_gpu)
        torch.cuda.synchronize()
        t_bwd_gpu = time.time() - t0
        print(f"    GPU反向投影耗时: {t_bwd_gpu:.3f}s (CPU: {t_bwd:.3f}s)")

        # 验证CPU/GPU结果一致性
        sinogram_cpu_check = sinogram_gpu.cpu()
        max_diff = (sinogram_cpu_check - sinogram).abs().max().item()
        print(f"    CPU/GPU最大差异: {max_diff:.2e}")
    else:
        print("    CUDA不可用, 跳过GPU测试")

    print(f"\n[8] (B, 1, H, W) 输入格式测试:")
    phantom_c = phantom.unsqueeze(1)  # (1, 1, H, W)
    sino_c = radon.forward(phantom_c)
    bp_c = radon.backward(sino_c)
    print(f"    输入 (1, 1, H, W) -> forward -> {sino_c.shape}")
    print(f"    sinogram -> backward -> {bp_c.shape}")
    assert sino_c.shape == (1, n_angles, img_size), "带通道维度的前向形状错误!"

    print(f"\n[9] 角度约定验证:")
    # 用一个偏离中心的点源测试角度方向
    point = torch.zeros(1, img_size, img_size)
    point[0, img_size // 4, img_size // 2] = 1.0  # 点在图像上半部分
    sino_point = radon.forward(point)
    # 在 angle=0 时, 点源应该出现在探测器的中间位置
    # 在 angle=pi/2 时, 投影应该沿着x方向积分
    idx_0 = 0
    idx_90 = n_angles // 4  # approximately pi/2
    pos_0 = sino_point[0, idx_0, :].argmax().item()
    pos_90 = sino_point[0, idx_90, :].argmax().item()
    print(f"    点源位置: (row={img_size // 4}, col={img_size // 2})")
    print(f"    angle=0 时投影峰值位置: {pos_0} (期望接近 {img_size // 2})")
    print(f"    angle=pi/2 时投影峰值位置: {pos_90} (期望接近 {img_size // 4})")

    print(f"\n{'=' * 60}")
    print("所有测试通过!")
    print(f"{'=' * 60}")
