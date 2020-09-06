import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import cluster.utils.utils_cluster as utils_cluster


def mass_center(mask):
    # calculate mass center from top-left corner
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
    print("Reading ground truth CSV")
    ground_truth_csv = pd.read_csv('single_instance_dataset_wradius.csv')
    csv = {
        'model_name': [],
        'mask_name': [],
        'mask_x': [],
        'mask_y': [],
        'component_x': [],
        'component_y': [],
        'pixel_distance': [],
        'norm_distance': [],
        'prediction_area': [],
        'gt_area': [],
        'component_area': [],
        'component_relative_area': [],
        'component_intersection': [],
        'component_union': [],
        'component_iou': [],
        'component_pw_recall': [],
        'component_pw_precision': [],
    }
    GT_IMAGES = os.path.join('.', 'images', 'masks_resize')
    MODELS_DIR = os.path.join('.', 'output', 'validation')
    
    model_list = [ # Models to be evaluated
        '11FCMN8rmsprop_lr0.0001_prep_keras_dp0.001_ep200',
        '11FCMN16rmsprop_lr0.0001_prep_keras_dp0.001_ep200',
        '11FCMN32rmsprop_lr0.0001_prep_keras_dp0.001_ep200'
    ]
    for model in model_list:
        print(model)
        model_name = model
        mask_list = os.listdir(
            os.path.join(MODELS_DIR, model, 'prediction_masks'))
        for mask in mask_list:

            row = utils_cluster.get_sample_ground_truth(mask, ground_truth_csv)
            gt_center = np.ndarray([1, 2])
            gt_center[0, 0] = (row['x_center_resize'].values[0])/2
            gt_center[0, 1] = (row['y_center_resize'].values[0])/2
            diam_resize = (row['diam_resize'].values[0])/2

            mask_generated = (utils_cluster.read_image_grayscale(
                os.path.join(
                    MODELS_DIR, model, 'prediction_masks', mask)))
            mask_generated = (mask_generated > 100).astype(int)
            ground_truth = (utils_cluster.read_image_grayscale(
                os.path.join(
                    GT_IMAGES, 'mask_'+mask[3:-3]+'png')))
            ground_truth = cv2.resize(ground_truth, (0, 0), fx=0.5, fy=0.5)
            ground_truth = (ground_truth > 0).astype(int)

            num_labels, labeled_img, centers, iou_array = connected_components_with_threshold(
            (mask_generated).astype(np.uint8), 0, ground_truth)

            gt_area = np.sum(ground_truth)
            prediction_area = np.sum(mask_generated.astype(bool))
            # computo de metricas segun resultado de component analysis

            if centers is not None:
                for component_label in range(num_labels):
                    filtered_component = labeled_img == (component_label + 1)
                    component_center = mass_center(filtered_component)

                    pixel_distance = np.linalg.norm(
                        np.subtract(gt_center, component_center))

                    csv['model_name'].append(model_name)
                    csv['mask_name'].append(mask)

                    csv['mask_x'].append(gt_center[0, 0])
                    csv['mask_y'].append(gt_center[0, 1])
                    # component_x
                    csv['component_x'].append(component_center[0])
                    # component_y
                    csv['component_y'].append(component_center[1])
                    csv['pixel_distance'].append(pixel_distance)
                    csv['norm_distance'].append(pixel_distance / diam_resize)
                    csv['gt_area'].append(gt_area)
                    csv['prediction_area'].append(prediction_area)
                    # component_area
                    component_area = np.sum(filtered_component)
                    csv['component_area'].append(component_area)
                    # relative_area
                    csv['component_relative_area'].append(
                        component_area / prediction_area)
                    # component pixelwise metrics
                    intersection = np.sum(np.logical_and(filtered_component, ground_truth))
                    union = np.sum(np.logical_or(filtered_component, ground_truth))
                    csv['component_intersection'].append(intersection)
                    csv['component_union'].append(union)
                    csv['component_iou'].append(intersection / union)
                    csv['component_pw_recall'].append(intersection / gt_area)
                    csv['component_pw_precision'].append(intersection / component_area)

            else:  # no buds detected register it in the metrics dict
                csv['model_name'].append(model_name)
                csv['mask_name'].append(mask)

                csv['mask_x'].append(gt_center[0, 0])
                csv['mask_y'].append(gt_center[0, 1])
                # component_x
                csv['component_x'].append(np.nan)
                # component_y
                csv['component_y'].append(np.nan)
                csv['gt_area'].append(gt_area)
                csv['pixel_distance'].append(np.nan)
                csv['norm_distance'].append(np.nan)

                csv['prediction_area'].append(np.nan)
                # component_area
                csv['component_area'].append(np.nan)
                # relative_area
                csv['component_relative_area'].append(np.nan)

                csv['component_intersection'].append(np.nan)
                csv['component_union'].append(np.nan)
                csv['component_iou'].append(np.nan)
                csv['component_pw_recall'].append(np.nan)
                csv['component_pw_precision'].append(np.nan)

    csv = pd.DataFrame(csv)
    csv.to_csv('full_component_report_fcn.csv')
