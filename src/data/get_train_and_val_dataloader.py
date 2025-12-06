import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os
from monai.transforms import MapTransform
def get_data_dicts(ids_path: str, shuffle: bool = False, first_n=False):

    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep=",")
    if shuffle:
        df = df.sample(frac=1, random_state=1)
    df = list(df)
    data_dicts = []
    for row in df:
        data_dicts.append({"image": (row)})
    if first_n is not False:
        data_dicts = data_dicts[:first_n]

    print(f"Found {len(data_dicts)} subjects.")
    if dist.is_initialized():
        print(dist.get_rank())
        print(dist.get_world_size())
        return partition_dataset(
            data=data_dicts,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
    else:
        return data_dicts
def get_data_dict_from_txt(ids_path: str, shuffle: bool = False, first_n=False, dataset = 'C16', val = True):
    paths_train=[]
    labels_train=[]
    if dataset=='TCGA_Gastric':
        loaded_nums = np.loadtxt("/data/output/patch/TCGA-Gastric/TCGA-Gastric_224_compression_name/train_wsi_nums.txt", dtype=int)
        train_num, val_num = train_test_split(loaded_nums, test_size=0.1764, random_state=42)
        paths = []
        labels = []
        wsi_nums = []
        with open(ids_path, "r") as file:
            for line in file:
                path, label,wsi_num = line.strip().split(' ')
                label=int(label)
                wsi_num= int(wsi_num)
                if label in [0,1,2]:
                    paths.append(path)
                    labels.append(label)
                    wsi_nums.append(wsi_num)
        wsi_nums = np.array(wsi_nums)
        if  'TCGA-Gastric_224_compression_name/patch_path_test.txt' in ids_path:
            paths_train = paths
        
        else:
            for item, wsi_number in enumerate(wsi_nums):
                if val:
                    if wsi_number in val_num:
                        paths_train.append(paths[item])
                        labels_train.append(labels[item])
                else:
                    if wsi_number not in val_num:
                        paths_train.append(paths[item])
                        labels_train.append(labels[item])
    elif dataset=='TCGA_rare_gastric':
        paths = []
        labels = []
        wsi_nums = []
        for data_path in ids_path.split(","):
            with open(data_path,"r") as f:
                lines_train = f.readlines()
            for index, line in enumerate(lines_train):
                path = line.split(' ')[0].split()[0]
                label=line.split(' ')[1].split()[0]
                label =int(label)
                if label in [3,4]:
                    paths_train.append(path)
                    labels.append(label)
    elif dataset=='Imagenet':
        paths_train = []
        for dirpath, dirnames, filenames in os.walk(ids_path):
            for file in filenames:
                paths_train.append(os.path.join(dirpath, file))
    elif dataset in ['ssb_hard','TCGA_lung']:
        with open(ids_path, "r") as file:
            for line in file:
                path, label = line.strip().split(' ')
                paths_train.append(path)
    elif dataset in ['TCGA_breast','RCC']:
        with open(ids_path, "r") as file:
            for line in file:
                paths_train.append(line.strip())
    else:
        with open(ids_path, "r") as file:
            for line in file:
                if dataset=='C16':
                    #print(line)
                    path_ = line.strip().split(' ')
                    if isinstance(path_, list) and len(path_) == 1:
                        path, label = line.strip().split(';')
                    else:
                        path = path_[0]
                else:
                    path = line.strip().split(' ')
                paths_train.append(path)

    #paths_train = np.array(paths_train)
    #labels_train = np.array(labels_train)
    data_dict = []
    for row in paths_train:
        data_dict.append(row)
    if shuffle:
        random.shuffle(data_dicts)

    data_dicts = [{"image": image_path, "path": image_path} for image_path in data_dict]
    if first_n is not False:
        data_dicts = data_dicts[:first_n]
    #data_dicts = [{"image": image_path, "path": image_path} for image_path in image_paths]
    #print(data_dicts[:5])
    return data_dicts

def get_training_data_loader(
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    only_val: bool = False,
    augmentation: bool = True,
    drop_last: bool = False,
    num_workers: int = 8,
    num_val_workers: int = 3,
    cache_data=True,
    first_n=None,
    is_grayscale=False,
    add_vflip=False,
    add_hflip=False,
    image_size=None,
    image_roi=None,
    spatial_dimension=2,
    dataset = 'C16'
):
    # Define transformations
    resize_transform = (
        transforms.ResizeD(keys=["image"], spatial_size=(image_size,) * spatial_dimension)
        if image_size
        else lambda x: x
    )

    central_crop_transform = (
        transforms.CenterSpatialCropD(keys=["image"], roi_size=image_roi)
        if image_roi
        else lambda x: x
    )
    class LoadNumpyd(MapTransform):
        """
        自定义 MONAI 变换：用于加载 .npy 格式的 NumPy 数组
        """
        def __init__(self, keys):
            super().__init__(keys)

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                d[key] = np.load(d[key])  # 读取 .npy 文件
            return d
    val_transforms = transforms.Compose(
        [
            LoadNumpyd(keys=["image"]),#transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim=0) if is_grayscale else lambda x: x,
            transforms.Lambdad(keys="image", func=lambda x: x[0, None, ...])
            if is_grayscale
            else lambda x: x,  # needed for BRATs data with 4 modalities in 1
            central_crop_transform,
            resize_transform,
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.RandFlipD(keys=["image"], spatial_axis=0, prob=1.0)
            if add_vflip
            else lambda x: x,
            transforms.RandFlipD(keys=["image"], spatial_axis=1, prob=1.0)
            if add_hflip
            else lambda x: x,
            transforms.ToTensord(keys=["image"]),
        ]
    )

    # no augmentation for now
    if augmentation:
        train_transforms = val_transforms
    else:
        train_transforms = val_transforms

    #val_dicts = get_data_dicts(validation_ids, shuffle=False, first_n=first_n)
    val_dicts = get_data_dict_from_txt(validation_ids, shuffle=False, first_n=first_n, dataset = dataset,val = True)
    if first_n:
        val_dicts = val_dicts[:first_n]

    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=val_transforms,
        )
    print(val_ds[0]["image"].shape)
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_val_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    if only_val:
        return val_loader

    #train_dicts = get_data_dicts(training_ids, shuffle=False, first_n=first_n)
    train_dicts = get_data_dict_from_txt(training_ids, shuffle=False, first_n=first_n, dataset = dataset,val = False)
    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    return train_loader, val_loader
