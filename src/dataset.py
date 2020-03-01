import os, sys, os.path as osp
this_path = osp.split(osp.abspath(__file__))[0]
sys.path += [osp.join(this_path, 'pytorch_utils')]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, random
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pytorch_utils
from functools import partial
from multiprocessing import Pool
import cv2
from torchsampler.imbalanced import ImbalancedDatasetSampler

def write_files_to_folder(new_folder, filepaths):
    pytorch_utils.mkdir_if_missing(new_folder)

    for filepath in tqdm(filepaths):
        img = cv2.imread(filepath, 0)
        img = 255 - img
        saveto = osp.join(new_folder, osp.basename(filepath))
        cv2.imwrite(saveto, img)


def arrange_dataset_images(root_path, new_path, test_files_ratio=0.2):
    # all chars available in the dataset
    ascii_chars = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
       'a','b','d','e','f','g','h','n','q','r','t']
    
    # code to char mapping
    ascii_code_to_char = {"%x"%(ord(char)): char for char in ascii_chars}  

    # make new destination dir
    train_folder = osp.join(new_path, 'train')
    test_folder = osp.join(new_path, 'test')
    pytorch_utils.mkdir_if_missing(train_folder)
    pytorch_utils.mkdir_if_missing(test_folder)
    
    def get_char_for_folder(folder_name):
        # remove _ if applicable
        char_ascii = folder_name.split("_")[0]
        assert char_ascii in ascii_code_to_char, "ascii code {} not found".format(char_ascii)
        return ascii_code_to_char[char_ascii]

    for folder in tqdm(os.listdir(root_path)):
        current_folder = osp.join(root_path, folder)
        if os.path.isdir(current_folder):
            char = get_char_for_folder(folder)
            print("preparing the character {}".format(char))
 
            # go through all the folders and collect image files
            all_img_files = []
            for subfolder in os.listdir(current_folder):
                current_subfolder = osp.join(current_folder, subfolder)
                if os.path.isdir(current_subfolder):
                    all_img_files += [
                        osp.join(current_subfolder, filename) for filename in os.listdir(current_subfolder)
                                                                if filename.endswith("png")
                    ]

            random.shuffle(all_img_files)
            test_files_count = int(len(all_img_files) * test_files_ratio)
            train_files, test_files = all_img_files[test_files_count:], all_img_files[:test_files_count]
            print("total files {}, train {}, test {}".format(len(all_img_files), len(train_files), len(test_files)))

            # collect train, test files
            char_train_folder = osp.join(train_folder, char)
            char_test_folder = osp.join(test_folder, char)
            write_files_to_folder(char_train_folder, train_files)
            write_files_to_folder(char_test_folder, test_files)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def download_and_arrange_data():
    # download data
    url = "https://s3.amazonaws.com/nist-srd/SD19/by_merge.zip"
    target_zipfile = '../data/emnist.zip'
    extraction_dir = '../data/emnist_raw'
    dataset_folder = '../data/emnist'
    #pytorch_utils.download_url(url, target_zipfile)
    #pytorch_utils.unzip_file(target_zipfile, extraction_dir)

    # arrange the data folders
    # arrange_dataset_images(osp.join(extraction_dir, "by_merge"), dataset_folder)
    return dataset_folder


def get_dataloaders():
    # download dataset
    root = download_and_arrange_data()

    # transforms for train and test
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ImageFolder(root=osp.join(root, 'train'), transform=train_transform, loader=pil_loader)
    test_dataset = ImageFolder(root=osp.join(root, 'test'), transform=test_transform, loader=pil_loader)
    train_loader = DataLoader(dataset=train_dataset, batch_size=192, 
                                drop_last=True, sampler=ImbalancedDatasetSampler(train_dataset))
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=64)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()   

    for data, target in train_loader:
        print(data.shape, target.shape)
        break

    for data, target in test_loader:
        print(data.shape, target.shape)
        break

    