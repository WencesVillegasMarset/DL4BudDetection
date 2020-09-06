import matplotlib.pyplot as plt
import json
from utils import utils_cluster
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import faulthandler
from datetime import datetime

faulthandler.enable()

def mass_center(mask):
    '''
        Calculate mass center from top-left corner
    '''
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0, mask.shape[0]):
        x_by_mass += np.sum(x * mask[:, x])
        y_by_mass += np.sum(x * mask[x, :])

    return (x_by_mass/total_mass, y_by_mass/total_mass)


def connected_components_with_threshold(image, threshold, ground_truth):
    '''
        Function that takes a mask and filters its component given a provided threshold
        this returns the number of resulting components and a new filtered mask (tuple) 
    '''

    num_components, mask = cv2.connectedComponents(image)
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    component_list = []
    mass_center_array = []
    iou_array = []

    for component in np.arange(1, num_components):
        isolated_component = (mask == component)
        if np.sum(isolated_component) >= threshold:
            iou_array.append(np.sum(np.logical_and(isolated_component, ground_truth)
                                    ) / np.sum(np.logical_or(isolated_component, ground_truth)))
            mass_center_array.append([component, mass_center(
                isolated_component.astype(int))])
            filtered_mask += isolated_component.astype(np.uint8)*component
            component_list.append(component)
    if len(component_list) == 0:
        mass_center_array = None
        return len(component_list), filtered_mask, mass_center_array, np.asarray(iou_array)
    else:
        print((num_components))
        return len(component_list), filtered_mask, np.asarray(mass_center_array, dtype=object), np.asarray(iou_array)


if __name__ == "__main__":
    base_path = os.path.join('..')
    dataset_path = os.path.join(base_path, 'images')
    model_folder_name = "MODEL_FOLDER"

    print("Reading ground truth CSV")
    ground_truth_csv = pd.read_csv(os.path.join(
        base_path, 'single_instance_dataset_wradius.csv'))

    route_csv = pd.read_csv(os.path.join(
        base_path, 'cluster_route.csv'), header=None)
    base_images_path = os.path.join(base_path, "output", "validation",
                                    model_folder_name, "prediction_masks")

    model_validation_folder = os.path.split(base_images_path)[0]
    model_name = os.path.split(model_validation_folder)[1]
    print("Processing images in: " + base_images_path)
    img_list = sorted(os.listdir(base_images_path))
    metrics = {
        'model_name': [],
        'mask_name': [],
        'mask_x': [],
        'mask_y': [],
        'component_x': [],
        'component_y': [],
        'component_iou': [],
        'pixel_distance': [],
        'norm_distance': [],
        'prediction_area': [],
        'component_area': [],
        'relative_area': [],
    }

    for img in img_list:
        print('Processing :' + img)
        # cluster image and get a labeled image where each pixel has a label value and a number of labels (ignoring 0 and -1)
        #image_data = utils_cluster.preprocess_image(utils_cluster.read_image_grayscale(os.path.join(base_images_path, img)))
        image_data = (utils_cluster.read_image_grayscale(
            os.path.join(base_images_path, img)))
        # LOAD GROUND TRUTH MASK
        ground_truth = (utils_cluster.read_image_grayscale(
            os.path.join(dataset_path, 'masks_resize', 'mask_'+img[3:-3]+'png')))
        # resize mask to 1024x1024
        ground_truth = cv2.resize(ground_truth, (0, 0), fx=0.5, fy=0.5)

        # get ground truth data
        row = utils_cluster.get_sample_ground_truth(img, ground_truth_csv)
        gt_center = np.ndarray([1, 2])
        gt_center[0, 0] = (row['x_center_resize'].values[0])/2
        gt_center[0, 1] = (row['y_center_resize'].values[0])/2
        diam_resize = (row['diam_resize'].values[0])/2

        # prediction_area
        prediction_area = np.sum(image_data.astype(bool))

        num_labels, labeled_img, centers, iou_array = connected_components_with_threshold(
            (image_data > 100).astype(np.uint8), 0, ground_truth)

        if centers is not None:
            for component_label in range(num_labels):
                filtered_component = labeled_img == (component_label + 1)
                component_center = mass_center(filtered_component)

                pixel_distance = np.linalg.norm(
                    np.subtract(gt_center, component_center))

                metrics['model_name'].append(model_name)
                metrics['mask_name'].append(img)

                metrics['mask_x'].append(gt_center[0, 0])
                metrics['mask_y'].append(gt_center[0, 1])

                metrics['component_iou'].append(iou_array[component_label])
                # component_x
                metrics['component_x'].append(component_center[0])
                # component_y
                metrics['component_y'].append(component_center[1])

                metrics['pixel_distance'].append(pixel_distance)
                metrics['norm_distance'].append(pixel_distance / diam_resize)

                metrics['prediction_area'].append(prediction_area)
                # component_area
                component_area = np.sum(filtered_component)
                metrics['component_area'].append(component_area)
                # relative_area
                metrics['relative_area'].append(
                    component_area / prediction_area)

        else:  # no buds detected register it in the metrics dict
            metrics['model_name'].append(model_name)
            metrics['mask_name'].append(img)

            metrics['mask_x'].append(gt_center[0, 0])
            metrics['mask_y'].append(gt_center[0, 1])

            metrics['component_iou'].append(0)

            # component_x
            metrics['component_x'].append(np.nan)
            # component_y
            metrics['component_y'].append(np.nan)

            metrics['pixel_distance'].append(np.nan)
            metrics['norm_distance'].append(np.nan)

            metrics['prediction_area'].append(np.nan)
            # component_area
            metrics['component_area'].append(np.nan)
            # relative_area
            metrics['relative_area'].append(np.nan)

    data = pd.DataFrame(metrics)
    data.to_csv(os.path.join(model_validation_folder, str(datetime.timestamp(datetime.now())) +
                             '_component_mass_'+model_name+'.csv'))
