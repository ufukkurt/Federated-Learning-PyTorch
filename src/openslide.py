#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pandas as pd
import numpy as np
from os.path import join
import glob
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import sys
import fileinput


data_path = '/home/masterthesis/ufuk/content/drive/MyDrive/ICIAR2018_BACH_Challenge/Photos'
image_filenames = glob.glob(join(data_path, "**/*.tif"), recursive=True)

print(len(image_filenames))

#slide = slideio.open_slide('../0b1d9f15-e9ee-426d-bfdf-d4a945aa0cf8.svs','SVS')
#num_scenes = slide.num_scenes
#scene = slide.get_scene(0)
#print(num_scenes, scene.name, scene.rect, scene.num_channels)
#print(scene.rect)
#raw_string = slide.raw_metadata
#raw_string.split("|")
#print(raw_string)

#for channel in range(scene.num_channels):
    #print(scene.get_channel_data_type(channel))

#image = scene.read_block(scene.rect, size=(500,0))
#plt.imshow(image, cmap='gray')
#plt.show()
#im = Image.fromarray(image)
#im.save("asda.jpg")


#slide = slideio.open_slide('../0b1d9f15-e9ee-426d-bfdf-d4a945aa0cf8.svs','SVS')
#scene = slide.get_scene(0)
#image = scene.read_block(slices=(0,1), frames=(0,5))
#print(image.shape)

import gdown

