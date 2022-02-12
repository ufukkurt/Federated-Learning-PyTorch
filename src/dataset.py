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

    print("mean, std")
    print(mean, std)
    return mean, std


class TrainingDataset(Dataset):
    def __init__(self, benign_trainset_inds, insitu_trainset_inds, invasive_trainset_inds, normal_trainset_inds,
                 benign_image_filenames, insitu_image_filenames, invasive_image_filenames, normal_image_filenames):
        #self.data_frame = pd.read_csv(
        #    '/home/masterthesis/ufuk/content/drive/MyDrive/ICIAR2018_BACH_Challenge/labels.txt', header=None,
        #    names=['path', 'id'], delim_whitespace=True)
        self.data_frame = pd.read_csv(
            'D:\TUM\Tez\Federated-Learning-PyTorch\content\drive\MyDrive\ICIAR2018_BACH_Challenge\labels.txt', header=None,
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
        label = self.data_frame.loc[self.data_frame['path'] == mainImagePath.rsplit('\\', 1)[-1]]['id'].item()
        transform = transforms.ToTensor()
        grayImage = ImageOps.grayscale(image)

        return transform(image)



data_path = 'D:\TUM\Tez\Federated-Learning-PyTorch\content\drive\MyDrive\ICIAR2018_BACH_Challenge\Photos'
benign_data_path = 'D:\TUM\Tez\Federated-Learning-PyTorch\content\drive\MyDrive\ICIAR2018_BACH_Challenge\Photos\Benign'
insitu_data_path = 'D:\TUM\Tez\Federated-Learning-PyTorch\content\drive\MyDrive\ICIAR2018_BACH_Challenge\Photos\InSitu'
invasive_data_path = 'D:\TUM\Tez\Federated-Learning-PyTorch\content\drive\MyDrive\ICIAR2018_BACH_Challenge\Photos\Invasive'
normal_data_path = 'D:\TUM\Tez\Federated-Learning-PyTorch\content\drive\MyDrive\ICIAR2018_BACH_Challenge\Photos\\Normal'
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
dataiter = iter(loader)
image1 = dataiter.next()
print("image1")
print(image1)
print("image1")

mean, std = compute_mean_std(loader)



