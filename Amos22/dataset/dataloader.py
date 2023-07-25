import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
import SimpleITK as sitk
import json
import os
from random import random, sample, seed


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


class Amos2022Dataset(Dataset):
    def __init__(self, data_list, args, phase='train'):
        super(Amos2022Dataset, self).__init__()
        self.data_list = data_list

        self.train_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),

                transforms.SpatialPadd(["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                       mode='symmetric'),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=args.pos,
                    neg=args.neg,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),

                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.load_image(self.data_list[item])
        if self.phase == 'train':
            data = self.train_transform(data)
        elif self.phase == 'val':
            data = self.val_transform(data)
        elif self.phase == 'test':
            data = self.test_transform(data)

        return data

    def load_image(self, file_dic):
        image_path = file_dic['image']
        label_path = file_dic['label']

        img = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        img = sitk.GetArrayFromImage(img).astype(np.float_)
        label = sitk.GetArrayFromImage(label).astype(np.float_)

        img = np.moveaxis(img, (0, 1, 2), (2, 1, 0))
        label = np.moveaxis(label, (0, 1, 2), (2, 1, 0))
        return{
            'image': img,
            'label': label,
            'path': image_path
        }

def get_loader(datasets,
               datalist_json,
               data_dir,
               fold,
               batch_size,
               num_works,
               args=None,
               phase=None):
    """

    :param datasets:
    :param datalist_json:
    :param data_dir:
    :param fold:
    :param batch_size:
    :param num_works:
    :param args:
    :param phase: None or 'ssl' 预训练时传入字符串， datafold_read 函数跳过路径生成，直接读取 json 文件中路径
    :return:
    """

    train_files, validation_files = datafold_read(datalist=datalist_json,
                                                  basedir=data_dir,
                                                  fold=fold)

    seed(12)
    sample_num = np.ceil(len(train_files) * args.frac).astype(np.int_)
    train_files = sample(train_files, sample_num)
    
    if phase == 'test':
        test_datasets = eval(datasets)(data_list=validation_files, phase='test', args=args)
        test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 num_workers=num_works,
                                 pin_memory=True,
                                 shuffle=False)
        return test_dataloader
    else:
        train_datasets = eval(datasets)(data_list=train_files, phase='train', args=args)
        val_datasets = eval(datasets)(data_list=validation_files, phase='val', args=args)

        train_dataloader = DataLoader(train_datasets,
                                    batch_size=batch_size,
                                    num_workers=num_works,
                                    pin_memory=True,
                                    shuffle=True)
        # 数据集图像尺寸不一致，推断时使用 滑动窗口推断 输入全尺寸，batch_size 设置为 1 防止无法堆叠
        val_dataloader = DataLoader(val_datasets,
                                    batch_size=1,
                                    num_workers=num_works,
                                    pin_memory=True,
                                    shuffle=False)
        return train_dataloader, val_dataloader



