"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models





def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    # example_list = (('../input_images/snake.jpg', 56),
    #                 ('../input_images/cat_dog.png', 243),
    #                 ('../input_images/spider.png', 72))
    example_list = (
        ("/home/makerspace/mks_users_home/dspfinal/training_data/data/train/linshan/1_trim_1.wav", 0),
        ("/home/makerspace/mks_users_home/dspfinal/training_data/data/train/yun/1_1.wav",1)
    )
    # img_path = example_list[example_index][0]
    sound_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    # original_sound = Image.open(img_path).convert('RGB')
    # Process image
    # prep_img = preprocess_image(original_image)
    # Define model
    # pretrained_model = models.alexnet(pretrained=True)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export
            )
