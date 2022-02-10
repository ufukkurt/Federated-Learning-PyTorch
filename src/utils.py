#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import glob
from os.path import join
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageOps


def shareInds(filenames):
    val_ratio = 0.4
    test_ratio = 0.75
    n_samples = len(filenames)
    shuffled_indices = np.random.permutation(n_samples)
    validationset_inds = shuffled_indices[:int(n_samples * val_ratio)]
    trainingset_inds = shuffled_indices[int(n_samples * val_ratio):]
    testset_inds = validationset_inds[:int(n_samples * val_ratio * test_ratio)]
    validationset_inds = validationset_inds[int(n_samples * val_ratio * test_ratio):]
    return trainingset_inds, validationset_inds, testset_inds


def imagePaths(benignPaths, insituPaths, invasivePaths, normalPath, benignInds, insituInds, inasiveInds, normalInds):
    imagePaths = []
    for ind in benignInds:
        croppedImagePaths = glob.glob(join(benignPaths[ind][:-4], "**/*"), recursive=True)
        for path in croppedImagePaths:
            imagePaths.append(path)

    for ind in insituInds:
        croppedImagePaths = glob.glob(join(insituPaths[ind][:-4], "**/*"), recursive=True)
        for path in croppedImagePaths:
            imagePaths.append(path)

    for ind in inasiveInds:
        croppedImagePaths = glob.glob(join(invasivePaths[ind][:-4], "**/*"), recursive=True)
        for path in croppedImagePaths:
            imagePaths.append(path)

    for ind in normalInds:
        croppedImagePaths = glob.glob(join(normalPath[ind][:-4], "**/*"), recursive=True)
        for path in croppedImagePaths:
            imagePaths.append(path)
    return imagePaths


def get_ds(client_number):
    data_path = '/content/drive/MyDrive/ICIAR2018_BACH_Challenge/Photos'
    benign_data_path = '/content/drive/MyDrive/ICIAR2018_BACH_Challenge/Photos/Benign'
    insitu_data_path = '/content/drive/MyDrive/ICIAR2018_BACH_Challenge/Photos/InSitu'
    invasive_data_path = '/content/drive/MyDrive/ICIAR2018_BACH_Challenge/Photos/Invasive'
    normal_data_path = '/content/drive/MyDrive/ICIAR2018_BACH_Challenge/Photos/Normal'
    image_filenames = glob.glob(join(data_path, "**/*.tif"), recursive=True)

    benign_image_filenames = glob.glob(join(benign_data_path, "**/*.tif"), recursive=True)
    insitu_image_filenames = glob.glob(join(insitu_data_path, "**/*.tif"), recursive=True)
    invasive_image_filenames = glob.glob(join(invasive_data_path, "**/*.tif"), recursive=True)
    normal_image_filenames = glob.glob(join(normal_data_path, "**/*.tif"), recursive=True)

    benign_trainset_inds, benign_valset_inds, benign_testset_inds = shareInds(benign_image_filenames)
    insitu_trainset_inds, insitu_valset_inds, insitu_testset_inds = shareInds(insitu_image_filenames)
    invasive_trainset_inds, invasive_valset_inds, invasive_testset_inds = shareInds(invasive_image_filenames)
    normal_trainset_inds, normal_valset_inds, normal_testset_inds = shareInds(normal_image_filenames)

    batch_size = 8

    mean_dataset = TrainingDataset(benign_trainset_inds, insitu_trainset_inds, invasive_trainset_inds,
                                   normal_trainset_inds,
                                   benign_image_filenames, insitu_image_filenames, invasive_image_filenames,
                                   normal_image_filenames)

    loader = torch.utils.data.DataLoader(dataset=mean_dataset,
                                         batch_size=batch_size,
                                         num_workers=0,
                                         shuffle=True, sampler=None,
                                         collate_fn=None)
    mean, std = compute_mean_std(loader)

    client_set_inds = []


    train_dataset = SlideTrainingDataset(benign_trainset_inds, insitu_trainset_inds, invasive_trainset_inds,
                                         normal_trainset_inds, benign_image_filenames, insitu_image_filenames, invasive_image_filenames,
                                         normal_image_filenames, mean, std)
    val_dataset = SlideValidationDataset(benign_valset_inds, insitu_valset_inds, invasive_valset_inds,
                                         normal_valset_inds, benign_image_filenames, insitu_image_filenames, invasive_image_filenames,
                                         normal_image_filenames, mean, std)
    test_dataset = TestDataset(benign_testset_inds, insitu_testset_inds, invasive_testset_inds, normal_testset_inds,
                               benign_image_filenames, insitu_image_filenames, invasive_image_filenames,
                               normal_image_filenames, mean, std)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True, sampler=None,
                                               collate_fn=None)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             shuffle=False, sampler=None,
                                             collate_fn=None)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              shuffle=False, sampler=None,
                                              collate_fn=None)

    return train_loader, val_loader, test_loader

def compute_mean_std(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


class TrainingDataset(Dataset):
    def __init__(self, benign_trainset_inds, insitu_trainset_inds, invasive_trainset_inds, normal_trainset_inds,
                 benign_image_filenames, insitu_image_filenames, invasive_image_filenames, normal_image_filenames):
        self.data_frame = pd.read_csv('/content/drive/MyDrive/ICIAR2018_BACH_Challenge/labels.txt', header=None,
                                      names=['path', 'id'], delim_whitespace=True)
        self.benignInds = benign_trainset_inds.tolist()
        self.insituInds = insitu_trainset_inds.tolist()
        self.invasiveInds = invasive_trainset_inds.tolist()
        self.normalInds = normal_trainset_inds.tolist()
        self.benign_image_filenames = benign_image_filenames
        self.insitu_image_filenames = insitu_image_filenames
        self.invasive_image_filenames = invasive_image_filenames
        self.normal_image_filenames = normal_image_filenames
        self.imagePaths = []

        self.imagePaths = imagePaths(benign_image_filenames, insitu_image_filenames, invasive_image_filenames,
                                     normal_image_filenames, self.benignInds, self.insituInds, self.invasiveInds,
                                     self.normalInds)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = Image.open(self.imagePaths[idx])
        mainImagePath = self.imagePaths[idx][:-3] + '.tif'
        label = self.data_frame.loc[self.data_frame['path'] == mainImagePath.rsplit('/', 1)[-1]]['id'].item()
        transform = transforms.ToTensor()
        grayImage = ImageOps.grayscale(image)

        return transform(image)


class SlideTrainingDataset(Dataset):
    def __init__(self, inds1, inds2, inds3, inds4, benign_image_filenames, insitu_image_filenames, invasive_image_filenames
                 , normal_image_filenames, mean, std):
        self.data_frame = pd.read_csv('/content/drive/MyDrive/ICIAR2018_BACH_Challenge/labels.txt', header=None,
                                      names=['path', 'id'], delim_whitespace=True)
        self.benignInds = inds1
        self.insituInds = inds2
        self.invasiveInds = inds3
        self.normalInds = inds4
        self.benign_image_filenames = benign_image_filenames
        self.insitu_image_filenames = insitu_image_filenames
        self.invasive_image_filenames = invasive_image_filenames
        self.normal_image_filenames = normal_image_filenames
        self.mean = mean
        self.std = std

        self.imagePaths = []

        self.imagePaths = imagePaths(self.benign_image_filenames, self.insitu_image_filenames,
                                     self.invasive_image_filenames, self.normal_image_filenames,
                                     self.benignInds, self.insituInds, self.invasiveInds, self.normalInds)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = Image.open(self.imagePaths[idx])
        mainImagePath = self.imagePaths[idx][:-3] + '.tif'
        label = self.data_frame.loc[self.data_frame['path'] == mainImagePath.rsplit('/', 1)[-1]]['id'].item()
        labelArray = [0.0, 0.0, 0.0, 0.0]
        labelArray[label] = 1.0
        # resized_image = torch.nn.functional.interpolate(image, size=(512, 512), mode='bilinear')
        # transform_norm = transforms.ToTensor()

        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.mean, self.std)])

        return transform_norm(image), torch.tensor(labelArray)


class SlideValidationDataset(Dataset):
    def __init__(self, inds1, inds2, inds3, inds4, benign_image_filenames, insitu_image_filenames, invasive_image_filenames
                 , normal_image_filenames, mean, std):
        self.data_frame = pd.read_csv('/content/drive/MyDrive/ICIAR2018_BACH_Challenge/labels.txt', header=None,
                                      names=['path', 'id'], delim_whitespace=True)
        self.benignInds = inds1
        self.insituInds = inds2
        self.invasiveInds = inds3
        self.normalInds = inds4
        self.benign_image_filenames = benign_image_filenames
        self.insitu_image_filenames = insitu_image_filenames
        self.invasive_image_filenames = invasive_image_filenames
        self.normal_image_filenames = normal_image_filenames
        self.mean = mean
        self.std = std
        self.imagePaths = []

        self.imagePaths = imagePaths(self.benign_image_filenames, self.insitu_image_filenames,
                                     self.invasive_image_filenames, self.normal_image_filenames,
                                     self.benignInds, self.insituInds, self.invasiveInds, self.normalInds)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = Image.open(self.imagePaths[idx])
        mainImagePath = self.imagePaths[idx][:-3] + '.tif'
        label = self.data_frame.loc[self.data_frame['path'] == mainImagePath.rsplit('/', 1)[-1]]['id'].item()
        labelArray = [0.0, 0.0, 0.0, 0.0]
        labelArray[label] = 1.0
        # transform_norm = transforms.ToTensor()

        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.mean, self.std)])

        return transform_norm(image), torch.tensor(labelArray)


def imagePathsTest(benignPaths, insituPaths, invasivePaths, normalPath, benignInds, insituInds, inasiveInds,
                   normalInds):
    imagePaths = []
    index = benignInds.tolist() + insituInds.tolist() + inasiveInds.tolist() + normalInds.tolist()
    image_filenames = glob.glob(join(data_path, "**/*.tif"), recursive=True)
    for indx in index:
        imagePaths.append(image_filenames[indx])

    return imagePaths


class TestDataset(Dataset):
    def __init__(self, inds1, inds2, inds3, inds4, benign_image_filenames, insitu_image_filenames, invasive_image_filenames
                 , normal_image_filenames, mean, std):
        self.data_frame = pd.read_csv('/content/drive/MyDrive/ICIAR2018_BACH_Challenge/labels.txt', header=None,
                                      names=['path', 'id'], delim_whitespace=True)
        self.benignInds = inds1
        self.insituInds = inds2
        self.invasiveInds = inds3
        self.normalInds = inds4
        self.benign_image_filenames = benign_image_filenames
        self.insitu_image_filenames = insitu_image_filenames
        self.invasive_image_filenames = invasive_image_filenames
        self.normal_image_filenames = normal_image_filenames
        self.mean = mean
        self.std = std
        self.imagePaths = []

        self.imagePaths = imagePaths(self.benign_image_filenames, self.insitu_image_filenames,
                                     self.invasive_image_filenames, self.normal_image_filenames,
                                     self.benignInds, self.insituInds, self.invasiveInds, self.normalInds)

        self.imagePathsTest = imagePathsTest(benign_image_filenames, insitu_image_filenames, invasive_image_filenames,
                                             normal_image_filenames,
                                             self.benignInds, self.insituInds, self.invasiveInds, self.normalInds)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = Image.open(self.imagePaths[idx])
        mainImagePath = self.imagePaths[idx][:-3] + '.tif'
        label = self.data_frame.loc[self.data_frame['path'] == mainImagePath.rsplit('/', 1)[-1]]['id'].item()
        # resized_image = torch.nn.functional.interpolate(image, size=(512, 512), mode='bilinear')
        image_size = (512, 512)
        resized_image = image.resize(image_size)
        labelArray = [0.0, 0.0, 0.0, 0.0]
        labelArray[label] = 1.0
        # transform_norm = transforms.ToTensor()

        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        image_folder = self.imagePaths[idx][:-3]
        image_folder = image_folder[-4:]

        return transform_norm(resized_image), torch.tensor(labelArray), image_folder


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
