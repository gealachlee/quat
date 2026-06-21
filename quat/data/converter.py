import numpy as np


class RealToQuaternionConverter:
    """将实数向量/图像转换为四元数
    
    对于彩色图像 (CIFAR-10等):
        q = 0 + R*i + G*j + B*k
        - 实部(q1) = 0
        - 虚部i(q2) = R通道
        - 虚部j(q3) = G通道
        - 虚部k(q4) = B通道
        
    对于灰度图像或其他数据:
        将数据分成4个通道，不足则padding
    """

    def __init__(self, n_channels=4, is_color_image=False):
        self.n_channels = n_channels
        self.is_color_image = is_color_image

    def convert(self, x):
        """将单个实数向量/图像转换为四元数
        
        Args:
            x: 实数向量或图像 (d,) 或 (h, w, c)
        Returns:
            Quaternion对象或四元数向量
        """
        if self.is_color_image and len(x.shape) == 3:
            return self._convert_image(x)
        else:
            return self._convert_vector(x)

    def _convert_image(self, img):
        """将彩色图像转换为四元数
        
        CIFAR-10: 32x32x3 -> q = 0 + R*i + G*j + B*k
        
        Args:
            img: (h, w, 3) 彩色图像
        Returns:
            Quaternion: 四元数 (实部=0, 虚部=R,G,B)
        """
        if img.shape[2] != 3:
            raise ValueError("Color image must have 3 channels (RGB)")
        
        R = img[:, :, 0].flatten()
        G = img[:, :, 1].flatten()
        B = img[:, :, 2].flatten()
        
        h, w = img.shape[:2]
        
        return {
            'real': np.zeros(h * w),
            'i': R,
            'j': G,
            'k': B,
            'shape': (h, w)
        }

    def _convert_vector(self, x):
        """将实数向量转换为四元数向量
        
        Args:
            x: (d,) 实数向量
        Returns:
            (n_channels, dim_per_channel) 四元数矩阵
        """
        d = len(x)
        dim_per_channel = (d + self.n_channels - 1) // self.n_channels
        
        result = np.zeros((self.n_channels, dim_per_channel))
        
        for i in range(self.n_channels):
            start_idx = i * dim_per_channel
            end_idx = min(start_idx + dim_per_channel, d)
            if start_idx < d:
                channel_data = x[start_idx:end_idx]
                result[i, :len(channel_data)] = channel_data
        
        return result

    def convert_batch(self, X):
        """批量转换
        
        Args:
            X: (n_samples, d) 或 (n_samples, h, w, c)
        Returns:
            四元数数据
        """
        if self.is_color_image and len(X.shape) == 4 and X.shape[3] == 3:
            return self._convert_batch_images(X)
        else:
            return self._convert_batch_vectors(X)

    def _convert_batch_images(self, images):
        """批量转换彩色图像
        
        Args:
            images: (n_samples, h, w, 3)
        Returns:
            dict with 'real', 'i', 'j', 'k', 'shape'
        """
        n_samples = images.shape[0]
        
        h, w = images.shape[1:3]
        size = h * w
        
        images_norm = images.astype(np.float32) / 255.0
        
        real = np.zeros((n_samples, size))
        i_channel = images_norm[:, :, :, 0].reshape(n_samples, -1)
        j_channel = images_norm[:, :, :, 1].reshape(n_samples, -1)
        k_channel = images_norm[:, :, :, 2].reshape(n_samples, -1)
        
        return {
            'real': real,
            'i': i_channel,
            'j': j_channel,
            'k': k_channel,
            'shape': (h, w)
        }

    def _convert_batch_vectors(self, X):
        """批量转换实数向量矩阵"""
        n_samples = X.shape[0]
        dim_per_channel = (X.shape[1] + self.n_channels - 1) // self.n_channels
        
        result = np.zeros((n_samples, self.n_channels, dim_per_channel))
        
        for i in range(n_samples):
            result[i] = self._convert_vector(X[i])
        
        return result

    def get_output_dim(self, input_dim):
        """获取输出维度"""
        if self.is_color_image:
            h = int(np.sqrt(input_dim / 3))
            w = h
            return (h, w)
        else:
            dim_per_channel = (input_dim + self.n_channels - 1) // self.n_channels
            return (self.n_channels, dim_per_channel)


def convert_cifar10_to_quaternion(X, normalize='zscore'):
    """专门用于CIFAR-10的转换函数
    
    CIFAR-10图像 (n, 32, 32, 3) -> 四元数
    
    q = 0 + R*i + G*j + B*k
    
    Args:
        X: (n_samples, 32, 32, 3) CIFAR-10图像
        normalize: 标准化方式
            - 'zscore': z-score标准化 (默认)
            - 'minmax': 归一化到[0,1]
            - None: 不标准化
    Returns:
        dict with 'real', 'i', 'j', 'k' arrays
    """
    n_samples = X.shape[0]
    
    X_float = X.astype(np.float32)
    
    if normalize == 'zscore':
        mean = X_float.mean(axis=(1, 2, 3), keepdims=True)
        std = X_float.std(axis=(1, 2, 3), keepdims=True) + 1e-7
        X_norm = (X_float - mean) / std
    elif normalize == 'minmax':
        X_norm = X_float / 255.0
    else:
        X_norm = X_float
    
    real = np.zeros((n_samples, 32 * 32))  # real为0
    i_channel = X_norm[:, :, :, 0].reshape(n_samples, -1)  # R通道 -> i
    j_channel = X_norm[:, :, :, 1].reshape(n_samples, -1)  # G通道 -> j
    k_channel = X_norm[:, :, :, 2].reshape(n_samples, -1)  # B通道 -> k
    
    return {
        'real': real,
        'i': i_channel,
        'j': j_channel,
        'k': k_channel,
        'shape': (32, 32)
    }


def zscore_normalize(X):
    """Z-score标准化
    
    z = (x - mean) / std
    
    Args:
        X: (n_samples, ...) 输入数据
        
    Returns:
        标准化后的数据
    """
    X_float = X.astype(np.float32)
    
    if X.ndim == 4:
        mean = X_float.mean(axis=(1, 2, 3), keepdims=True)
        std = X_float.std(axis=(1, 2, 3), keepdims=True) + 1e-7
    elif X.ndim == 2:
        mean = X_float.mean(axis=1, keepdims=True)
        std = X_float.std(axis=1, keepdims=True) + 1e-7
    else:
        mean = X_float.mean()
        std = X_float.std() + 1e-7
    
    return (X_float - mean) / std


def quaternion_to_real(Q):
    """将四元数转换回实数"""
    return Q.flatten()


def quaternion_to_real_batch(Q_batch):
    """批量将四元数转换回实数"""
    return Q_batch.reshape(Q_batch.shape[0], -1)
