import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
import SimpleITK as sitk
import json
import os


def datafold_read(datalist, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


class BasicDataset(Dataset):
    def __init__(self, data_list, args, phase='train'):
        super(BasicDataset, self).__init__()
        self.data_list = data_list

        self.train_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image"]),
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
        return data

    def load_image(self, file_dic):
        image_path = file_dic['image']
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        image = np.moveaxis(image, (0, 1, 2), (2, 1, 0))
        return {'image': image,
                'image_path': image_path}


class PretrainDatasetCT(BasicDataset):
    def __init__(self, data_list, args, phase='train'):
        super(PretrainDatasetCT, self).__init__(data_list, args, phase='train')
        self.data_list = data_list

        self.train_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),

                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),

                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),

                # Make sure the two sampled 3D volumes are partial overlap
                transforms.SpatialPadd(keys="image",
                                       spatial_size=[args.roi_x + 32,
                                                     args.roi_y + 32,
                                                     args.roi_z + 32],
                                       mode='symmetric'),
                transforms.RandSpatialCropd(keys="image",
                                            roi_size=[args.roi_x + 32,
                                                      args.roi_y + 32,
                                                      args.roi_z + 32]),

                transforms.RandSpatialCropSamplesd(
                    keys=['image'],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),

                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),

                # Make sure the two sampled 3D volumes are partial overlap
                transforms.SpatialPadd(keys="image",
                                       spatial_size=[args.roi_x + 32,
                                                     args.roi_y + 32,
                                                     args.roi_z + 32],
                                       mode='symmetric'),
                transforms.RandSpatialCropd(keys="image",
                                            roi_size=[args.roi_x + 32,
                                                      args.roi_y + 32,
                                                      args.roi_z + 32]),

                transforms.RandSpatialCropSamplesd(
                    keys=['image'],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.phase = phase


class PretrainDatasetMR(BasicDataset):
    def __init__(self, data_list, args, phase='train'):
        super(PretrainDatasetMR, self).__init__(data_list, args, phase='train')

        self.train_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),

                transforms.SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z], mode='symmetric'),

                # Make sure the two sampled 3D volumes are partial overlap
                transforms.RandSpatialCropd(keys="image",
                                            roi_size=[args.roi_x + 32,
                                                      args.roi_y + 32,
                                                      args.roi_z]),

                transforms.RandSpatialCropSamplesd(
                    keys=['image'],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),

                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
                # Make sure the two sampled 3D volumes are partial overlap
                transforms.RandSpatialCropd(keys="image",
                                            roi_size=[args.roi_x + 32,
                                                      args.roi_y + 32,
                                                      args.roi_z]),
                transforms.RandSpatialCropSamplesd(
                    keys=['image'],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.phase = phase

    def load_image(self, file_dic):
        image = []
        assert len(file_dic['image']) == 4, f"BraTS need four modality but get {len(file_dic['image'])}"

        for i in sorted(file_dic['image']):
            image_path = os.path.join('/home/qlc/raid/dataset/BraTS2021', i)

            image_ = sitk.ReadImage(image_path)

            image_ = np.moveaxis(image_, (0, 1, 2), (2, 1, 0))

            image.append(image_)
        image = np.stack(image, axis=0)
        return {
            'image': image,
            'image_path': file_dic,
        }


def get_loader( datasets,
                datalist_json,
                fold,
                batch_size,
                num_works,
                args):
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
                                                  fold=fold)

    train_datasets = eval(datasets)(data_list=train_files, phase='train', args=args)
    val_datasets = eval(datasets)(data_list=validation_files, phase='val', args=args)

    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  num_workers=num_works,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=True)

    val_dataloader = DataLoader(val_datasets,
                                batch_size=batch_size,
                                num_workers=num_works,
                                pin_memory=True,
                                shuffle=True,
                                drop_last=True)

    return train_dataloader, val_dataloader



