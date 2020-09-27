from keras.utils import Sequence
import numpy as np
import os.path as path
# from skimage.io import imread
import cv2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

class DataGeneratorMobileNetKeras(Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=None,dim_label=None, n_channels=3,
                n_channels_label = 1, shuffle=True,img_path='/',mask_path='/', augmentation=True):
                self.dim = dim
                self.dim_label = dim_label
                self.batch_size = batch_size
                self.labels = labels
                self.n_channels_label = n_channels_label
                self.list_IDs = list_IDs
                self.n_channels = n_channels
                self.shuffle = shuffle
                self.on_epoch_end()
                self.img_path = img_path
                self.mask_path = mask_path
                self.augmentation = augmentation
                self.image_gen = ImageDataGenerator(rotation_range=45., width_shift_range=0.4, height_shift_range=0.4,
                                                    shear_range=0.1, zoom_range=0.3, horizontal_flip=True, vertical_flip=True,
                                                    fill_mode='constant', cval=0, dtype=np.float64)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, 1024, 1024, self.n_channels))
        y = np.empty((self.batch_size, 1024, 1024, self.n_channels_label))

        for index, id in enumerate(list_IDs_temp):
            #store image array
            seed = np.random.randint(10000000)
            temp = preprocess_input(image.img_to_array(image.load_img(path.join(self.img_path,id))))
            temp = cv2.resize(temp, (0, 0), fx=0.5, fy=0.5)

            if self.augmentation:
                temp  = self.image_gen.random_transform(temp, seed=seed)
            
            X[index,] = temp
            
            temp = image.img_to_array(image.load_img(path.join(self.mask_path,self.labels[id])))
            temp = cv2.resize(temp, (0, 0), fx=0.5, fy=0.5)
            if self.augmentation:
                temp  = self.image_gen.random_transform(temp, seed=seed)

            temp = cv2.cvtColor(temp,cv2.COLOR_RGB2GRAY).astype(bool).astype(np.float64)
            temp = np.expand_dims(temp, axis=2)
            y[index,] = temp
        return X, y

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y    