import cv2
import numpy as np 
import pandas as pd
import os

def get_sample_name(str):
    return (str.split('.')[0]).split('_')[1]

def remove_extension_from_filename(str):
    return (str.split('.')[0])

def filter_out_background_px(array):
    new_array = array[array[:,0] > 0]
    return new_array


def get_sample_ground_truth(sample_name, csv):
    name = (sample_name.split('.')[1][1::])+'.jpg'
    return (csv.loc[csv['imageOrigin'] == name, :])
    
    
def grayscale_to_rgb(grayscale):
    blank_ch = 255*np.ones_like(grayscale)
    labeled_img = cv2.merge([grayscale, blank_ch, blank_ch])
    return cv2.cvtColor(np.uint8(labeled_img), cv2.COLOR_HSV2RGB)


def read_image_grayscale(path):
    return cv2.imread(path, 0)


def save_image(image, out_path, image_name):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cv2.imwrite(os.path.join(out_path, image_name), image)

    
def mass_center(mask):
    #calculate mass center from top-left corner
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0,mask.shape[0]):
        x_by_mass += np.sum(x * mask[:,x])
        y_by_mass += np.sum(x * mask[x,:])

    return((x_by_mass/total_mass, y_by_mass/total_mass))

def preprocess_image(image):
    indices = np.dstack(np.indices(image.shape))
    xycolors = np.concatenate((np.expand_dims(image,axis=2), indices), axis=-1) 
    return np.reshape(xycolors, [-1,3])