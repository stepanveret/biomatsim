"""
Code is based on: https://github.com/AMLab-Amsterdam/DIVA
"""
import glob
from os import fspath
from re import I
from threading import Condition
import numpy.random
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
from PIL import Image

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.transforms.functional


class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = numpy.random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = torchvision.transforms.functional.rotate(img, random_rotation*90)
        return img


class ToyCell(data_utils.Dataset):    
    def __init__(self, path, data_info_filename, features_names, condition="*",  
                 img_list=None, transform=False, imgsize=(128, 128), scaler=True, 
                 filenumber_length=7, filter_config=None):
        """
        filter_config = {
            "roundness": [0, 10],
            "elongation": [0, 10] 
        }
        """
        self.path = path  # "synthetic_dataset/dataset2_128_fixloc_1scale/"
        self.data_info_filename = data_info_filename  # "dataset2_128_fixloc_1scale.csv"
        self.features_names = features_names  # ['roundness', 'elongation', 'nucleus_size', 'rotation_angle']
        self.condition = condition
        self.img_list = img_list
        self.scaler = scaler
        self.filenumber_length = filenumber_length
        self.filter_config = filter_config

        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.imgsize = imgsize
        self.resize = transforms.Resize(self.imgsize, interpolation=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.to_pil = transforms.ToPILImage()
        self.rrotate = RandomRotate()

        self.images, self.features_list = self.get_data()

    def get_img_tensor(self):
        tensor_list = []
        for filename in tqdm(self.filenames):
            with open(filename, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')
            tensor_list.append(self.to_tensor(self.resize(img)))

        # Concatenate
        return torch.stack(tensor_list)

    def get_data(self):

        # get filenames (filtered by a regex condition if there is one)
        filenames = [f for f in glob.glob(self.path + self.condition + ".png", recursive=True)]

        # sort filenames
        filenames = sorted(filenames, key = lambda x: x[-self.filenumber_length-4:]) # len(".png") == 4 #x.split("\\")[-1])
        self.filenames = filenames

        # load data info table
        data_info_table = pd.read_csv(self.path + self.data_info_filename, index_col=0)
        data_info_table['filename'] = data_info_table['idx'].apply(lambda x: str(x).zfill(7) + '.png')
        data_info_table.drop('idx', axis=1, inplace=True)

        if self.filter_config:
            for feature_name in self.filter_config.keys():
                low, high = self.filter_config[feature_name]
                data_info_table = data_info_table[(data_info_table[feature_name] > low) & (data_info_table[feature_name] < high)]

        # get min max values (before filtering by condition/img_list)
        dfmin = data_info_table[self.features_names].min()
        dfmax = data_info_table[self.features_names].max()

        # filter df filenames by self.condition
        self.condition_filenames = [filename[-self.filenumber_length-4:] for filename in self.filenames]
        data_info_table = data_info_table[data_info_table['filename'].isin(self.condition_filenames)]

        # filter df by img_list (train/val)
        if self.img_list:
            data_info_table = data_info_table[data_info_table['filename'].isin(self.img_list)]

        # sort df by filename
        data_info_table.sort_values(by='filename', inplace=True)

        # store data_info_table
        self.data_info_table = data_info_table
        df = data_info_table[self.features_names]

        # store normalized data_info_table
        if self.scaler:
            ndf = (df-dfmin)/(dfmax-dfmin)
        else:
            ndf = df

        self.ndf = ndf

        # create list of feature values
        features_list = [ndf[fn].values.reshape(-1, 1) for fn in self.features_names]  # [column1_data, column2_data, ]
        features_list = [torch.from_numpy(arr) for arr in features_list]

        # filter the initial list self.filenames after filtering df by img_list
        self.filenames = list(filter(lambda x: x[-self.filenumber_length-4:] in data_info_table['filename'].values, 
                                                                                                        self.filenames))
        # load images
        images = self.get_img_tensor()

        return images, features_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):  # index - a 1D array of batch_size
        x = self.images[index]
        fs = [arr[index] for arr in self.features_list]  # list of columns of batch_size

        if self.transform:
            x = self.to_tensor(self.rrotate(self.vflip(self.hflip(self.to_pil(x)))))

        return x, fs  # fs = list of arrays (columns) [(batch_size, 1), (batch_size, 1), ..]


class ToyCellpair(data_utils.Dataset):    
    def __init__(self, path_l, path_x, path_table, data_info_filename, 
                 features_names_l, features_names_x, condition_l="*", condition_x="*", 
                 img_list_l=None, img_list_x=None, imgsize=(128, 128), scaler=True, 
                 filenumber_length=7, filter_config=None):

        self.path_l = path_l  # "synthetic_dataset/pdata1_128/"
        self.path_x = path_x  # "synthetic_dataset/dataset2_128_fixloc_1scale/"
        self.path_table = path_table
        self.data_info_filename = data_info_filename  # "dataset2_128_fixloc_1scale.csv"
        self.features_names_l = features_names_l  # ['roundness_g1', 'radius_g2']
        self.features_names_x = features_names_x  # ['roundness_f1', 'elongation_f2', 'nucleus_size_f3', 'rotation_angle_f4']
        self.filter_config = filter_config

        self.condition_l = condition_l 
        self.condition_x = condition_x
        self.img_list_l = img_list_l
        self.img_list_x = img_list_x
        self.scaler = scaler
        self.filenumber_length = filenumber_length

        self.to_tensor = transforms.ToTensor()
        self.imgsize = imgsize
        self.resize = transforms.Resize(self.imgsize, interpolation=Image.BILINEAR)

        self.images_l, self.features_list_l,\
            self.images_x, self.features_list_x = self.get_data()

    def get_img_tensor(self, filenames):

        tensor_list = []
        for filename in tqdm(filenames):
            with open(filename, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')
            tensor_list.append(self.to_tensor(self.resize(img)))

        # Concatenate
        return torch.stack(tensor_list)

    def get_data(self):

        # get filenames l
        self.filenames_l = [f for f in glob.glob(self.path_l + self.condition_l + ".png", recursive=True)]
        self.condition_filenames_l = [filename[-self.filenumber_length-4:] for filename in self.filenames_l]
        self.filename_start_l = self.filenames_l[0][:-self.filenumber_length-4]

        # get filenames x (filtered by a regex condition if there is one)
        self.filenames_x = [f for f in glob.glob(self.path_x + self.condition_x + "*.png", recursive=True)]
        self.condition_filenames_x = [filename[-self.filenumber_length-4:] for filename in self.filenames_x]
        self.filename_start_x = self.filenames_x[0][:-self.filenumber_length-4]

        # load data info table
        data_info_table = pd.read_csv(self.path_table + self.data_info_filename, index_col=0)

        if self.filter_config:
            for feature_name in self.filter_config.keys():
                low, high = self.filter_config[feature_name]
                data_info_table = data_info_table[(data_info_table[feature_name] > low) & (data_info_table[feature_name] < high)]

        # get min max values (before filtering)
        dfmin = data_info_table[self.features_names_l + self.features_names_x].min()
        dfmax = data_info_table[self.features_names_l + self.features_names_x].max()

        # filter df by conditions
        data_info_table = data_info_table[data_info_table['filename_l'].isin(self.condition_filenames_l)]
        data_info_table = data_info_table[data_info_table['filename_x'].isin(self.condition_filenames_x)]

        # filter df by img_list (train/val)
        if self.img_list_x:
            data_info_table = data_info_table[data_info_table['filename_x'].isin(self.img_list_x)]
        if self.img_list_l:
            data_info_table = data_info_table[data_info_table['filename_l'].isin(self.img_list_l)]

        # sort df by filename_l (or x)
        data_info_table.sort_values(by='filename_l', inplace=True)  # !!!

        # store data_info_table
        self.data_info_table = data_info_table
        df = data_info_table[self.features_names_l + self.features_names_x]

        # store normalized data_info_table
        if self.scaler:
            ndf = (df-dfmin)/(dfmax-dfmin)
        else:
            ndf = df

        self.ndf = ndf

        # create lists of feature values for topographies/cells
        features_list_l = [self.ndf[fn].values.reshape(-1, 1) for fn in self.features_names_l]  # [column1_data, column2_data, ]
        features_list_l = [torch.from_numpy(arr) for arr in features_list_l]
        features_list_x = [self.ndf[fn].values.reshape(-1, 1) for fn in self.features_names_x]  # [column1_data, column2_data, ]
        features_list_x = [torch.from_numpy(arr) for arr in features_list_x]


        # create img file lists for topographies/cells
        # it's important to preserve order of pairs (topography, cell) according to the table in file lists
        self.filenames_l_column = data_info_table['filename_l'].values
        self.filenames_x_column = data_info_table['filename_x'].values
        self.final_filenames_l = [self.filename_start_l + f for f in self.filenames_l_column]
        self.final_filenames_x = [self.filename_start_x + f for f in self.filenames_x_column]

        # load images
        images_l = self.get_img_tensor(self.final_filenames_l)
        images_x = self.get_img_tensor(self.final_filenames_x)

        return images_l, features_list_l, images_x, features_list_x

    def __len__(self):
        return len(self.images_l)

    def __getitem__(self, index):  # index - a 1D array of batch_size

        l = self.images_l[index]
        x = self.images_x[index]

        fl = [arr[index] for arr in self.features_list_l]  # list of columns of batch_size
        fx = [arr[index] for arr in self.features_list_x]  # list of columns of batch_size

        return l, fl, x, fx
