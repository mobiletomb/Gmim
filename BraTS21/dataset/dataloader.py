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


class BraTsDataset(Dataset):
    def __init__(self, data_list, args, phase='train'):
        super(BraTsDataset, self).__init__()
        self.data_list = data_list

        self.train_transform = transforms.Compose(
            [
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.CropForegroundd(
                    keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
                ),
                transforms.RandSpatialCropd(
                    keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
        image = []
        assert len(file_dic['image']) == 4, f"BraTS need four modality but get {len(file_dic['image'])}"

        for i in sorted(file_dic['image']):
            image_path = i
            image_ = sitk.ReadImage(image_path)
            image_ = sitk.GetArrayFromImage(image_)
            image_ = np.moveaxis(image_, (0, 1, 2), (2, 1, 0))
            image.append(image_)
        image = np.stack(image, axis=0)

        label_path = file_dic['label']
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)
        label = np.moveaxis(label, (0, 1, 2), (2, 1, 0))

        return {
            'image': image,
            'label': label,
            'path': file_dic['image'][0]
        }
        

def get_loader(datasets,
               datalist_json,
               data_dir,
               fold,
               batch_size,
               num_works,
               args=None,
               phase=None):

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

        val_dataloader = DataLoader(val_datasets,
                                    batch_size=batch_size,
                                    num_workers=num_works,
                                    pin_memory=True,
                                    shuffle=False)
        return train_dataloader, val_dataloader





