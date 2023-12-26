"""
Upsample utils & layers
"""
from .utils import *
from .patch_utils import *


def crop_tensor_nthwc(x, size, contiguous=False):
    assert len(size) == 3
    if list(x.shape)[1:4] != size:
        crop_offset = _compute_cropping_offset(x.shape[1:4], size)
        x = x[:, crop_offset[0]:crop_offset[0] + size[0], crop_offset[1]:crop_offset[1] + size[1], crop_offset[2]:crop_offset[2] + size[2], :]
    if contiguous:
        x = x.contiguous()
    return x


def crop_tensor_ncthw(x, size, contiguous=False):
    assert len(size) == 3
    if list(x.shape)[2:5] != size:
        crop_offset = _compute_cropping_offset(x.shape[2:5], size)
        x = x[:, :, crop_offset[0]:crop_offset[0] + size[0], crop_offset[1]:crop_offset[1] + size[1], crop_offset[2]:crop_offset[2] + size[2]]
    if contiguous:
        x = x.contiguous()
    return x


def crop_tensor_nchw(x, size, contiguous=False):
    assert len(size) == 2
    if list(x.shape)[2:4] != size:
        crop_offset = _compute_cropping_offset(x.shape[2:4], size, 2)
        x = x[:, :, crop_offset[0]:crop_offset[0] + size[0], crop_offset[1]:crop_offset[1] + size[1]]
    if contiguous:
        x = x.contiguous()
    return x


def pad_tensor_nchw(x, size, contiguous=False, value=0.):
    assert len(size) == 2
    if list(x.shape)[2:4] != size:
        padding = _compute_padding_offset(x.shape[2:4], size, 2)
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]), value=value)
    if contiguous:
        x = x.contiguous()
    return x


def _compute_cropping_offset(size, target, ndims=3):
    return [(size[d] - target[d]) // 2 for d in range(ndims)]


def _compute_padding_offset(size, target, ndims=3):
    return _compute_cropping_offset(target, size, ndims)


def _interpolate(x, scale_factor, mode, align_corners):
    if x.ndim == 4:
        _, _, H, W = x.shape
        size = (scale_factor[0] * T, scale_factor[1] * H, scale_factor[2] * W)
    elif x.ndim == 5:
        _, _, T, H, W = x.shape
        size = (scale_factor[0] * T, scale_factor[1] * H, scale_factor[2] * W)
    else:
        raise NotImplementedError
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def _convert_index(x, in_size, out_size, align_corners: bool):
    # See https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/UpSample.h
    if align_corners:
        y = (out_size - 1.) / (in_size - 1.) * x
    else:
        y = out_size / in_size * (x + 0.5) - 0.5
    return y

# kernel_bicubic_alignfalse=[
#     [0.0012359619 ,0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 ,0.0037078857 ,0.0012359619],
#     [0.0037078857 ,0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 ,0.0111236572 ,0.0037078857],
#     [-0.0092010498 ,-0.0276031494 ,0.0684967041 ,0.2300262451 ,0.2300262451 ,0.0684967041 ,-0.0276031494 ,-0.0092010498],
#     [-0.0308990479 ,-0.0926971436 ,0.2300262451 ,0.7724761963 ,0.7724761963 ,0.2300262451 ,-0.0926971436 ,-0.0308990479],
#     [-0.0308990479 ,-0.0926971436 ,0.2300262451 ,0.7724761963 ,0.7724761963 ,0.2300262451 ,-0.0926971436 ,-0.0308990479],
#     [-0.0092010498 ,-0.0276031494 ,0.0684967041 ,0.2300262451 ,0.2300262451 ,0.0684967041 ,-0.0276031494 ,-0.0092010498],
#     [0.0037078857 ,0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 ,0.0111236572 ,0.0037078857],
#     [0.0012359619 ,0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 ,0.0037078857 ,0.0012359619],
# ]

def cubic_interpolation_kernel(x):
    abs_x = torch.abs(x)
    if abs_x <= 1:
        return (1.5 * abs_x - 2.5) * abs_x**2 + 1
    elif 1 < abs_x <= 2:
        return ((-0.5 * abs_x + 2.5) * abs_x - 4) * abs_x + 2
    else:
        return 0

def generate_cubic_interpolation_matrix(kernel_size, upsample_factor):
    center = (kernel_size - 1) / 2.0
    normal_scale = kernel_size / 2.0
    matrix = torch.zeros((kernel_size, kernel_size), dtype=torch.float16)
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = (j - center) / normal_scale
            y = (i - center) / normal_scale
            matrix[i, j] = cubic_interpolation_kernel(x) * cubic_interpolation_kernel(y)
    return matrix

class AdaptiveUpsampling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.method, self.align_corners = self.get_upsampling_options(cfg)
        self.patch_spec = None
        K = cfg['upsample_kernel_size']
        S = cfg['upsample_scale']
        C1 = cfg['C1']
        C2 = cfg['C2']

        assert C1 == C2
        self.upsampling_padding = (K//2, K//2, K//2, K//2)
        self.upsampling_crop=(S*K+K-S)//2
        upsampling_kernel = generate_cubic_interpolation_matrix(K,S)
        upsampling_kernel = torch.unsqueeze(upsampling_kernel, (0, 1))
        upsampling_kernel = upsampling_kernel.expand(-1,C2,-1,-1)
        self.upsampling_layer = nn.ConvTranspose2d(C1,C2,upsampling_kernel.shape,bias=False,stride=2,padding=self.upsampling_crop,group=C1)
        self.upsampling_layer.weight.data = nn.Parameter(upsampling_kernel,requires_grad=True)
    
    def extra_repr(self):
        s = 'method={method}, align_corners={align_corners}'
        return s.format(**self.__dict__)

    def get_upsampling_options(self, cfg):
        upsample_options = cfg.split(',')
        assert len(upsample_options) <= 2
        assert len(upsample_options) < 2 or 'align' in upsample_options
        method = upsample_options[0]
        align_corners = 'align' in upsample_options
        return method, align_corners

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int],
                patch_mode: bool=True):
        """
        Inputs:
            x: input tensor with shape [N, T1, H1, W1, C]
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes.
            size: list of 3 ints. Represents the size of the fulle video. It does not have to be the same as the input size, as the input can be a patch from the full video.
            scale: list of 3 ints. Represents the scale factor. This will be used to compute the output size.
            padding: list of 3 ints. Represents the padding size. This will be used to compute the output size.
            patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.

        Output:
            a tensor with shape [N, T2, H2, W2, C]
        """
                    
        x_pad = F.pad(x, self.upsampling_padding,mode='replicate')
        y_conv= self.upsampling_layer(x_pad)
        
        return upsampled_latent.permute((1,0,2,3))

class FastTrilinearInterpolation(nn.Module):
    """
    A module for switching implementaion of the trilinear interpolation.
    It also combines the interpolation with the cropping.
    """
    def __init__(self, cfg):
        super().__init__()
        self.method, self.align_corners = self.get_upsampling_options(cfg)
        self.patch_spec = None

    def extra_repr(self):
        s = 'method={method}, align_corners={align_corners}'
        return s.format(**self.__dict__)

    def get_upsampling_options(self, cfg):
        upsample_options = cfg.split(',')
        assert len(upsample_options) <= 2
        assert len(upsample_options) < 2 or 'align' in upsample_options
        method = upsample_options[0]
        align_corners = 'align' in upsample_options
        return method, align_corners

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int],
                patch_mode: bool=True):
        """
        Inputs:
            x: input tensor with shape [N, T1, H1, W1, C]
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes.
            size: list of 3 ints. Represents the size of the fulle video. It does not have to be the same as the input size, as the input can be a patch from the full video.
            scale: list of 3 ints. Represents the scale factor. This will be used to compute the output size.
            padding: list of 3 ints. Represents the padding size. This will be used to compute the output size.
            patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.

        Output:
            a tensor with shape [N, T2, H2, W2, C]
        """
        assert x.ndim == 5
        assert all(size[d] % scale[d] == 0 for d in range(3))
        in_sizes = [size[d] // scale[d] for d in range(3)]
        out_sizes = size
        in_patch_sizes = x.shape[1:4]
        out_patch_size = [out_sizes[d] // idx_max[d] + 2 * padding[d] for d in range(3)]
        in_padding = [(in_patch_sizes[d] - in_sizes[d] // idx_max[d]) // 2 for d in range(3)]
        out_padding = padding

        N, T1, H1, W1, C = x.shape
        T2, H2, W2 = out_patch_size

        method = self.method if patch_mode else 'interpolate'

        if method == 'interpolate':
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = _interpolate(x, scale_factor=scale, mode='trilinear', align_corners=self.align_corners)
            x = x.permute(0, 2, 3, 4, 1)
            x = crop_tensor_nthwc(x, out_patch_size).contiguous()
        elif method in ['matmul', 'matmul-th-w', 'matmul-t-h-w']:
            idx_in, _ = compute_pixel_idx_3d(idx, idx_max, in_sizes, in_padding, clipped=False)
            idx_out, idx_out_mask = compute_pixel_idx_3d(idx, idx_max, out_sizes, out_padding, clipped=False)

            idx_out_p = [_convert_index(idx_out[d], out_sizes[d], in_sizes[d], align_corners=self.align_corners).clip_(0, in_sizes[d] - 1) for d in range(3)]
            diff = [torch.abs(idx_out_p[d][:, :, None] - idx_in[d][:, None, :]) for d in range(3)]
            weights = [(1. - diff[d]) * (diff[d] <= 1.) * idx_out_mask[d][:, :, None] for d in range(3)]

            if method == 'matmul':
                M = (weights[0][:, :, None, None, :, None, None] * weights[1][:, None, :, None, None, :, None] * weights[2][:, None, None, :, None, None, :]).view(N, T2 * H2 * W2, T1 * H1 * W1)
                x = torch.matmul(M, x.view(N, T1 * H1 * W1, C)).view(N, T2, H2, W2, C)
                x = x.view(N, T2, H2, W2, C)
            elif method == 'matmul-th-w':
                M1 = (weights[0][:, :, None, :, None] * weights[1][:, None, :, None, :]).view(N, 1, T2 * H2, T1 * H1)
                M2 = weights[2].view(N, 1, W2, W1)
                x = torch.matmul(M1, x.view(N, 1, T1 * H1, W1 * C))
                x = torch.matmul(M2, x.view(N, T2 * H2, W1, C))
                x = x.view(N, T2, H2, W2, C)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return x


class FastNearestInterpolation(nn.Module):
    """
    A module for switching implementaion of the nearest interpolation. See FastTrilinearInterpolation for more details.
    """    
    def __init__(self, cfg):
        super().__init__()
        self.method = self.get_upsampling_options(cfg)
        self.patch_spec = None

    def extra_repr(self):
        s = 'method={method}'
        return s.format(**self.__dict__)

    def get_upsampling_options(self, cfg):
        upsample_options = cfg.split(',')
        assert len(upsample_options) <= 2
        assert len(upsample_options) < 2 or 'align' in upsample_options
        method = upsample_options[0]
        return method

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int],
                patch_mode: bool=True):
        assert x.ndim == 5
        assert all(size[d] % scale[d] == 0 for d in range(3))
        in_sizes = tuple(size[d] // scale[d] for d in range(3))
        out_sizes = size
        in_patch_sizes = tuple(x.shape[1:4])
        out_patch_size = tuple(out_sizes[d] // idx_max[d] + 2 * padding[d] for d in range(3))
        in_padding = tuple((in_patch_sizes[d] - in_sizes[d] // idx_max[d]) // 2 for d in range(3))
        out_padding = tuple(padding)

        N, T1, H1, W1, C = x.shape
        T2, H2, W2 = out_patch_size

        method = self.method if patch_mode else 'interpolate'

        if method == 'interpolate':
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = F.interpolate(x, scale_factor=scale, mode='nearest')
            x = x.permute(0, 2, 3, 4, 1)
            x = crop_tensor_nthwc(x, out_patch_size).contiguous()
        else:
            raise NotImplementedError
        return x
