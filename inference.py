import click
import pandas as pd
import os
import cv2
import numpy as np
import models
from utils import DataGeneratorMobileNetKeras
import re

@click.command()
@click.option('--model', required=True, help='Model name in output/models folder e.g model.h5')
@click.option('--output', default=os.path.join('.','output', 'validation'), help='Path to which the generated masks are written')
@click.option('--csv', default="./test.csv", help='Test set csv with (image, mask) tuples')
def inference(model, output, csv):
    np.random.seed(0)
    test_set_full = pd.read_csv(csv)
    list_ids = list(test_set_full['image'].values)
    list_masks = list(test_set_full['mask'].values)
    # get root directories
    img_path = os.path.split(list_ids[0])[0]
    mask_path = os.path.split(list_masks[0])[0]

    list_ids = [os.path.basename(img) for img in list_ids]
    list_masks = [os.path.basename(mask) for mask in list_masks]
    labels = dict(zip(list_ids, list_masks))


    mask_output_path = os.path.join(
        output, model, 'prediction_masks')
    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)

    valid_generator = DataGeneratorMobileNetKeras(batch_size=1, img_path=img_path, labels=labels,
                                                list_IDs=list_ids, n_channels=3, n_channels_label=1, shuffle=False, mask_path=mask_path, augmentation=False)
   
    model_loaded = models.load_model(os.path.join(
        '.', 'output', 'models', model))
    prediction = model_loaded.predict_generator(
        generator=valid_generator, use_multiprocessing=True, verbose=True)

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_images = list_ids
    for threshold in threshold_list:  
        array_pred = np.copy(prediction)
        for i in np.arange(0,prediction.shape[0]):
            # get prediction and normalize
            pred = array_pred[i, :, :, 0]
            pred = (pred > threshold).astype(bool)
            cv2.imwrite(mask_output_path + '/'+(str(threshold))+test_images[i], (pred.astype(np.uint8))*255)
    print(model + ' images generated!')


if __name__ == "__main__":
    inference()