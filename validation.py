import pandas as pd
import os
import cv2
import numpy as np
import models
from utils import DataGeneratorMobileNetKeras, DataGeneratorMobileNet
import re


def mass_center(mask):
    # calculate mass center from top-left corner
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0, mask.shape[0]):
        x_by_mass += np.sum(x * mask[:, x])
        y_by_mass += np.sum(x * mask[x, :])

    return((x_by_mass/total_mass, y_by_mass/total_mass))


def connected_components_with_threshold(image, threshold):
    '''
        Function that takes a mask and filters its component given a provided threshold
        this returns the number of resulting components and a new filtered mask (tuple) 
    '''
    num_components, mask = cv2.connectedComponents(image)
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    component_list = []
    mass_center_array = []

    for component in np.arange(1, num_components):
        isolated_component = (mask == component)
        if np.sum(isolated_component) >= threshold:
            mass_center_array.append(mass_center(
                isolated_component.astype(int)))
            filtered_mask += isolated_component
            component_list.append(component)
    if len(component_list) == 0:
        mass_center_array = np.nan
    return len(component_list), filtered_mask, (np.asarray(mass_center_array))


# compute validation metrics
def validate(**kwargs):

    if kwargs['preprocessing'] == True:
        valid_generator = DataGeneratorMobileNetKeras(batch_size=1, img_path=kwargs['img_path'], labels=kwargs['labels'],
                                                      list_IDs=kwargs['partition']['valid'], n_channels=3, n_channels_label=1, shuffle=False, mask_path=kwargs['masks_path'])
    else:
        valid_generator = DataGeneratorMobileNet(batch_size=1, img_path=kwargs['img_path'], labels=kwargs['labels'],
                                                 list_IDs=kwargs['partition']['valid'], n_channels=3, n_channels_label=1, shuffle=False, mask_path=kwargs['masks_path'])

    model = models.load_model(os.path.join(
        '.', 'output', 'models', kwargs['model_name']+'.h5'))
    prediction = model.predict_generator(
        generator=valid_generator, use_multiprocessing=True, workers=6, verbose=True)

    mask_output_path = os.path.join(
        kwargs['validation_folder'], kwargs['model_name'], 'prediction_masks')
    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_images = kwargs['partition']['valid']
    # labels = kwargs['labels']
    for threshold in threshold_list:  
        array_pred = np.copy(prediction)
        for i in np.arange(0,prediction.shape[0]):
            # get prediction and normalize
            pred = array_pred[i, :, :, 0]
            pred = (pred > threshold).astype(bool)
            cv2.imwrite(mask_output_path + '/'+(str(threshold))+test_images[i], (pred.astype(np.uint8))*255)
    print(kwargs['model_name'] + ' images generated!')


if __name__ == "__main__":
    list_models = pd.read_csv('models_to_validate.csv', header=None)
    list_models = list_models.iloc[:, 0].values
    test_set = pd.read_csv('single_instance_test.csv')
    test_set_array = test_set['imageOrigin'].values
    args = {}
    for model in list_models:
        args['preprocessing'] = True

        args['partition'] = {
            'train': [],
            'valid': test_set_array
        }

        args['labels'] = dict(
            zip(list(test_set['imageOrigin'].values), list(test_set['mask'].values)))
        args['model_name'] = model
        args['img_path'] = os.path.join('.', 'images', 'images_resize')
        args['masks_path'] = os.path.join('.', 'images', 'masks_resize')
        args['validation_folder'] = os.path.join('.', 'output', 'validation')
        validate(**args)
