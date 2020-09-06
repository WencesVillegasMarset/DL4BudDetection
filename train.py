import click
import os
import pandas as pd
import numpy as np
from utils import DataGeneratorMobileNetKeras
from keras.optimizers import RMSprop
from models import mobilenet_8s, mobilenet_16s, mobilenet_32s
from datetime import datetime

@click.command()
@click.option('--epochs', default=1, help='Number of epochs')
@click.option('--lr', default=0.0001, help='Learning rate')
@click.option('--bs', default=4, help='Batch size')
@click.option('--savemodel', default=True, is_flag=True, help='Save to trained model weights')
@click.option('--csv', default="./train.csv", help='Training set csv with (image, mask) tuples')
def train(epochs, lr, bs, savemodel, csv):
    np.random.seed(0)
    train_set_full = pd.read_csv(csv)
    list_ids = list(train_set_full['image'].values)
    list_masks = list(train_set_full['mask'].values)
    # get root directories
    img_path = os.path.split(list_ids[0])[0]
    mask_path = os.path.split(list_masks[0])[0]

    list_ids = [os.path.basename(img) for img in list_ids]
    list_masks = [os.path.basename(mask) for mask in list_masks]
    labels = dict(zip(list_ids, list_masks))

    out_models = os.path.join('.','output', 'models')
    out_history = os.path.join('.','output', 'history')
    if savemodel and not os.path.exists(out_models):
        os.makedirs(out_models)
    if not os.path.exists(out_history):
        os.makedirs(out_history) 

    for arch in [8, 16, 32]:
        print(f"Starting FCN-MN {arch}s training")
        if arch == 8:
            model = mobilenet_8s(train_encoder=True, final_layer_activation="sigmoid", prep=True)
        if arch == 16:
            model = mobilenet_16s(train_encoder=True, final_layer_activation="sigmoid", prep=True)
        else:
            model = mobilenet_32s(train_encoder=True, final_layer_activation="sigmoid", prep=True)

        train_generator = DataGeneratorMobileNetKeras(batch_size=bs, img_path=img_path,
                                        labels=labels, list_IDs=list_ids, n_channels=3,
                                        n_channels_label=1, shuffle=True, mask_path=mask_path)

        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

        train_history = model.fit_generator(generator=train_generator, use_multiprocessing=True, epochs=epochs)

        timestamp = str(datetime.now().strftime("%Y%m%d_%H-%M-%S"))

        model_name = "{}-{}s_fcn_mn".format(timestamp, arch)

        if savemodel:
            model.save(out_models, model_name + '.h5'))

        history_csv = pd.DataFrame(train_history.history)
        history_csv.to_csv(os.path.join(out_history, model_name +'.csv'))


if __name__ == '__main__':
    train()