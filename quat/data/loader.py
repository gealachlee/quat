import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore', message='.*dtype.*')
warnings.filterwarnings('ignore', message='.*align.*')

# 项目根目录的data文件夹
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def _get_data_path(*paths):
    """获取数据路径，优先使用BASE_DATA_DIR"""
    target = os.path.join(BASE_DATA_DIR, *paths) if paths else BASE_DATA_DIR
    if os.path.exists(target):
        return target
    # 尝试不带前缀
    return os.path.join(*paths) if paths else BASE_DATA_DIR


def _convert_to_quaternion_dict(X):
    """将图像数据转换为四元数dict格式（含/255归一化，用于原始uint8数据）

    Args:
        X: (n, h, w, 3) 图像数组

    Returns:
        X_dict: {'real': (n, h, w), 'i': (n, h, w), 'j': (n, h, w), 'k': (n, h, w)}
    """
    n = X.shape[0]
    X = X.astype(np.float32) / 255.0

    return {
        'real': np.zeros((n, X.shape[1], X.shape[2])),
        'i': X[:, :, :, 0],
        'j': X[:, :, :, 1],
        'k': X[:, :, :, 2]
    }


def _convert_to_quaternion_dict_no_norm(X):
    """将已标准化的图像数据转换为四元数dict格式（不做/255归一化）

    Args:
        X: (n, h, w, 3) 图像数组（已标准化，如StandardScaler后）

    Returns:
        X_dict: {'real': (n, h, w), 'i': (n, h, w), 'j': (n, h, w), 'k': (n, h, w)}
    """
    n = X.shape[0]
    X_f32 = X.astype(np.float32)

    return {
        'real': np.zeros((n, X_f32.shape[1], X_f32.shape[2])),
        'i': X_f32[:, :, :, 0],
        'j': X_f32[:, :, :, 1],
        'k': X_f32[:, :, :, 2]
    }


def _split_data(X, y, test_ratio=0.2, random_state=42):
    """分割训练集和测试集

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n = len(y)
    n_test = int(n * test_ratio)
    indices = np.random.permutation(n)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if isinstance(X, dict):
        X_train = {k: v[train_indices] for k, v in X.items()}
        X_test = {k: v[test_indices] for k, v in X.items()}
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    return X_train, X_test, y[train_indices], y[test_indices]


def load_cifar10_samples(n_per_class=1000, k=2, random_state=42, return_quaternion=False, test_ratio=0.3):
    """加载CIFAR-10数据集并采样
    
    Args:
        n_per_class: 每类采样数量
        k: 类别数量（必须是偶数）
        random_state: 随机种子
        return_quaternion: True返回四元数字典，False返回矩阵
        test_ratio: 测试集比例

    Returns:
        如果return_quaternion=False:
            X: (n_samples, h, w, 3) 图像数组
            y: (n_samples,) 标签
        如果return_quaternion=True:
            X_dict: {'real': real, 'i': i, 'j': j, 'k': k}
            y: (n_samples,) 标签
        训练集+测试集返回(X_train, y_train, X_test, y_test)
    """
    if k % 2 != 0:
        raise ValueError("k must be even")
    
    np.random.seed(random_state)
    
    cifar_dir = _get_data_path('cifar-10-python', 'cifar-10-batches-py')

    if os.path.exists(cifar_dir):
        train_data_list = []
        train_labels_list = []

        for i in range(1, 6):
            batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
            train_data_list.append(batch['data'])
            train_labels_list.append(batch['labels'])

        train_data = np.vstack(train_data_list)

        #
        # scaler = StandardScaler()
        # train_data = scaler.fit_transform(train_data)




        train_labels = np.hstack(train_labels_list)

        available_classes = list(range(10))
        selected_classes = available_classes[:k]

        X_list = []
        y_list = []

        for idx, cls in enumerate(selected_classes):
            cls_indices = np.where(train_labels == cls)[0]
            selected_indices = np.random.choice(
                cls_indices, size=min(n_per_class, len(cls_indices)), replace=False
            )
            X_list.append(train_data[selected_indices].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            y_list.append(np.full(len(selected_indices), idx))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
    else:
        raise FileNotFoundError(f"CIFAR-10 data not found in {cifar_dir}")

    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    X_train, X_test, y_train, y_test = _split_data(X, y, test_ratio, random_state)

    if return_quaternion:
        X_train = _convert_to_quaternion_dict(X_train)
        X_test = _convert_to_quaternion_dict(X_test)
        if return_quaternion == 'tensor':
            from quat.collections import dict_to_quat_tensor, labels_to_quat_vector
            X_train = dict_to_quat_tensor(X_train)
            X_test = dict_to_quat_tensor(X_test)
            y_train = labels_to_quat_vector(y_train)
            y_test = labels_to_quat_vector(y_test)
        return X_train, y_train, X_test, y_test

    return X_train, y_train, X_test, y_test


def load_stl10_samples(n_per_class=1000, k=2, random_state=42, return_quaternion=False, test_ratio=0.2):
    """加载STL-10数据集并采样（二分类）

    Args:
        n_per_class: 每类采样数量
        k: 类别数量（必须是偶数）
        random_state: 随机种子
        return_quaternion: True返回四元数字典，False返回矩阵
        test_ratio: 测试集比例

    Returns:
        如果return_quaternion=False:
            X: (n_samples, 96, 96, 3) 图像数组
            y: (n_samples,) 标签
        如果return_quaternion=True:
            X_dict: {'real': real, 'i': i, 'j': j, 'k': k}
            y: (n_samples,) 标签
        训练集+测试集返回(X_train, y_train, X_test, y_test)
    """
    if k % 2 != 0:
        raise ValueError("k must be even")

    np.random.seed(random_state)

    stl10_dir = _get_data_path('stl10_binary')

    if not os.path.exists(stl10_dir):
        raise FileNotFoundError(f"STL-10 data not found in {stl10_dir}")
    else:
        train_X = np.fromfile(os.path.join(stl10_dir, "train_X.bin"), dtype=np.uint8)
        train_y = np.fromfile(os.path.join(stl10_dir, "train_y.bin"), dtype=np.uint8)

        train_X = train_X.reshape(96, 96, 3, -1).transpose(3, 0, 1, 2)
        train_y = train_y - 1

        selected_classes = list(range(k))

        X_list = []
        y_list = []

        for idx, cls in enumerate(selected_classes):
            cls_indices = np.where(train_y == cls)[0]
            if len(cls_indices) == 0:
                continue
            selected_indices = np.random.choice(
                cls_indices, size=min(n_per_class, len(cls_indices)), replace=False
            )
            X_list.append(train_X[selected_indices])
            y_list.append(np.full(len(selected_indices), idx))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    X_train, X_test, y_train, y_test = _split_data(X, y, test_ratio, random_state)

    if return_quaternion:
        X_train = _convert_to_quaternion_dict(X_train)
        X_test = _convert_to_quaternion_dict(X_test)
        if return_quaternion == 'tensor':
            from quat.collections import dict_to_quat_tensor, labels_to_quat_vector
            X_train = dict_to_quat_tensor(X_train)
            X_test = dict_to_quat_tensor(X_test)
            y_train = labels_to_quat_vector(y_train)
            y_test = labels_to_quat_vector(y_test)
        return X_train, y_train, X_test, y_test

    return X_train, y_train, X_test, y_test


def load_flower17_samples(n_per_class=1000, k=2, random_state=42, return_quaternion=False, test_ratio=0.2):
    """加载Oxford Flowers-17数据集并采样
    
    Args:
        n_per_class: 每类采样数量
        k: 类别数量（必须是偶数）
        random_state: 随机种子
        return_quaternion: True返回四元数字典，False返回矩阵
        test_ratio: 测试集比例
    
    Returns:
        如果return_quaternion=False:
            X: (n_samples, 64, 64, 3) 图像数组
            y: (n_samples,) 标签
        如果return_quaternion=True:
            X_dict: {'real': real, 'i': i, 'j': j, 'k': k}
            y: (n_samples,) 标签
        训练集+测试集返回(X_train, y_train, X_test, y_test)
    """
    if k % 2 != 0:
        raise ValueError("k must be even")
    
    np.random.seed(random_state)
    
    flower17_dir = _get_data_path('17flowers', 'jpg')
    
    if not os.path.exists(flower17_dir):
        raise FileNotFoundError(f"Flowers-17 data not found in {flower17_dir}")
    
    jpg_files = sorted([f for f in os.listdir(flower17_dir) if f.lower().endswith('.jpg')])
    
    if len(jpg_files) < 136:
        raise FileNotFoundError(f"Insufficient Flowers-17 images found in {flower17_dir}")

    n_classes = 17
    images_per_class = 80  # Flowers-17 has exactly 80 images per class

    X_list = []
    y_list = []

    from PIL import Image

    for cls in range(min(k, n_classes)):
        # Randomly sample n_per_class images from the 80 available
        n_select = min(n_per_class, images_per_class)
        chosen_indices = np.random.choice(images_per_class, size=n_select, replace=False)

        for img_idx in chosen_indices:
            img_num = cls * images_per_class + img_idx + 1
            img_name = f"image_{img_num:04d}.jpg"
            img_path = os.path.join(flower17_dir, img_name)

            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
                X_list.append(np.array(img))
                y_list.append(cls)

    X = np.stack(X_list)
    y = np.array(y_list)

    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    X_train, X_test, y_train, y_test = _split_data(X, y, test_ratio, random_state)

    if return_quaternion:
        X_train = _convert_to_quaternion_dict(X_train)
        X_test = _convert_to_quaternion_dict(X_test)
        if return_quaternion == 'tensor':
            from quat.collections import dict_to_quat_tensor, labels_to_quat_vector
            X_train = dict_to_quat_tensor(X_train)
            X_test = dict_to_quat_tensor(X_test)
            y_train = labels_to_quat_vector(y_train)
            y_test = labels_to_quat_vector(y_test)
        return X_train, y_train, X_test, y_test

    return X_train, y_train, X_test, y_test


def load_svhn_samples(n_per_class=1000, k=2, random_state=42, return_quaternion=False, test_ratio=0.4, mat_file='train_32x32.mat'):
    """从 train_32x32.mat 加载SVHN数据集并采样

    SVHN (Street View House Numbers) 数据集包含32×32的彩色数字图像（0-9）。
    原始.mat格式:
        X: (32, 32, 3, n) uint8 — 图像数据
        y: (n, 1) — 标签 (1-10, 其中10代表数字0)

    预处理流程:
        1. 加载数据 → /255 归一化到[0,1]
        2. 从标签1开始按类别采样（k=2时取数字1和2）
        3. 可选转换为四元数格式

    四元数编码 (与converter.py一致):
        q = 0 + R*i + G*j + B*k

    Args:
        n_per_class: 每类采样数量
        k: 类别数量（必须为偶数，最大10）
        random_state: 随机种子
        return_quaternion: True返回四元数字典，False返回矩阵
        test_ratio: 测试集比例
        mat_file: .mat文件名

    Returns:
        如果return_quaternion=False:
            X: (n_samples, 32, 32, 3) 图像数组 (float32, /255归一化)
            y: (n_samples,) 标签 (0..k-1)
        如果return_quaternion=True:
            X_dict: {'real': real, 'i': i, 'j': j, 'k': k}
            y: (n_samples,) 标签 (0..k-1)
        训练集+测试集返回(X_train, y_train, X_test, y_test)
    """
    if k % 2 != 0:
        raise ValueError("k must be even")

    np.random.seed(random_state)

    mat_path = _get_data_path(mat_file)

    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"SVHN .mat file not found at {mat_path}. "
            f"Please download train_32x32.mat from http://ufldl.stanford.edu/housenumbers/ "
            f"and place it in the data/ directory."
        )

    # 加载.mat文件
    import scipy.io as sio
    data = sio.loadmat(mat_path)

    # SVHN格式: X shape (32, 32, 3, n), y shape (n, 1)
    X_raw = data['X']  # (32, 32, 3, n_samples) uint8
    y_raw = data['y']  # (n_samples, 1)

    # 转置为 (n_samples, 32, 32, 3)
    X = X_raw.transpose(3, 0, 1, 2).astype(np.float32)
    y = y_raw.flatten().astype(np.int64)

    # 仅做 /255 归一化，与 CIFAR-10 一致。
    # 不做 StandardScaler：逐像素维度零均值化会破坏四元数内积的类别判别力。
    X = X / 255.0

    # SVHN原始标签1-10（10=数字0），从1开始选取（数字1,2,3,...）
    available_classes = sorted(np.unique(y).tolist())  # [1, 2, ..., 10]
    selected_classes = available_classes[:k]  # [1, 2] when k=2

    X_list = []
    y_list = []

    for idx, cls in enumerate(selected_classes):
        cls_indices = np.where(y == cls)[0]
        n_available = len(cls_indices)
        n_select = min(n_per_class, n_available)
        selected_indices = np.random.choice(
            cls_indices, size=n_select, replace=False
        )
        X_list.append(X[selected_indices])
        # 重新映射标签为0..k-1
        y_list.append(np.full(n_select, idx))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 打乱
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    X_train, X_test, y_train, y_test = _split_data(X, y, test_ratio, random_state)

    if return_quaternion:
        X_train = _convert_to_quaternion_dict(X_train)
        X_test = _convert_to_quaternion_dict(X_test)
        if return_quaternion == 'tensor':
            from quat.collections import dict_to_quat_tensor, labels_to_quat_vector
            X_train = dict_to_quat_tensor(X_train)
            X_test = dict_to_quat_tensor(X_test)
            y_train = labels_to_quat_vector(y_train)
            y_test = labels_to_quat_vector(y_test)
        return X_train, y_train, X_test, y_test

    return X_train, y_train, X_test, y_test


def load_imagenette_samples(n_per_class=1000, k=2, random_state=42, return_quaternion=False, test_ratio=0.2, split='train', data_dir=None, img_size=(160, 160)):
    """加载ImageNette数据集并采样

    Args:
        n_per_class: 每类采样数量
        k: 类别数量
        random_state: 随机种子
        return_quaternion: True返回四元数字典，False返回矩阵
        test_ratio: 测试集比例
        split: 'train' or 'val'
        data_dir: 数据目录路径
        img_size: 目标图像大小
    
    Returns:
        如果return_quaternion=False:
            X: (n_samples, h, w, 3) 图像数组
            y: (n_samples,) 标签
        如果return_quaternion=True:
            X_dict: {'real': real, 'i': i, 'j': j, 'k': k}
            y: (n_samples,) 标签
        训练集+测试集返回(X_train, y_train, X_test, y_test)
    """
    if data_dir is None:
        data_dir = _get_data_path('imagenette2-160')

    np.random.seed(random_state)

    data_path = os.path.join(data_dir, split)

    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} does not exist")

    all_classes = sorted([d for d in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, d))])

    classes = all_classes[:k]

    X_list = []
    y_list = []

    for idx, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            continue

        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if n_per_class is not None:
            image_files = image_files[:n_per_class]

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')

                if img.size != img_size:
                    img = img.resize(img_size, Image.Resampling.LANCZOS)

                img_array = np.array(img, dtype=np.float32)
                img_array = img_array / 255.0

                X_list.append(img_array)
                y_list.append(idx)

            except Exception as e:
                continue

    if len(X_list) == 0:
        raise ValueError("No images were loaded successfully")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    
    X_train, X_test, y_train, y_test = _split_data(X, y, test_ratio, random_state)
    
    if return_quaternion:
        X_train = _convert_to_quaternion_dict(X_train)
        X_test = _convert_to_quaternion_dict(X_test)
        if return_quaternion == 'tensor':
            from quat.collections import dict_to_quat_tensor, labels_to_quat_vector
            X_train = dict_to_quat_tensor(X_train)
            X_test = dict_to_quat_tensor(X_test)
            y_train = labels_to_quat_vector(y_train)
            y_test = labels_to_quat_vector(y_test)
        return X_train, y_train, X_test, y_test
    
    return X_train, y_train, X_test, y_test
