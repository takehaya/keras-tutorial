"""
https://arakan-pgm-ai.hatenablog.com/entry/2018/10/24/090000
"""
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img

class NncData:
    def get_csv_list(self, fname):
        csv_file = open(fname, "r", encoding="utf8", errors="", newline="" )
        dset = csv.reader(
            csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True
        )
        next(dset)
        return dset

    def load_nnc_image(self, fname, gray=True, width=28, height=28, category=10, test=0.3, seed=14365):
        tx = []
        ty = []
        dcsv = self.get_csv_list(fname)
        for r in dcsv:
            if gray:
                temp_img = load_img("tools" + r[0].lstrip("."), color_mode="grayscale", target_size=(height, width))
            else:
                temp_img = load_img("tools" + r[0].lstrip("."), target_size=(height, width))
            img_array = img_to_array(temp_img)
            tx.append(img_array)
            ty.append(r[1])
        x_data = np.asarray(tx)
        y_label = np.asarray(ty)
        x_data = x_data.astype('float32')
        x_data = x_data / 255.0
        y_label = y_label.astype('float32')

        # y_label = np_utils.to_categorical(y_label, category)
        y_label = NncData.to_categorical(y_label, category)

        return train_test_split(x_data, y_label, test_size=test, random_state=seed)

    @staticmethod
    def to_categorical(y, nb_classes=None):
        '''Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy.
        '''
        if not nb_classes:
            nb_classes = np.max(y) + 1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, int(y[i])] = 1.
        return Y