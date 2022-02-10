#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

import slideio
import openslide
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2

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

url = r'gdown https://drive.google.com/uc?id=1-03IllgbzEF8Q00f5c4mBaMpLNBjpBKs'
gdown.download(url)