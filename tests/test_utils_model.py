import os
from shutil import copyfile

import numpy as np

from networkml.utils.model import Model

model = Model(10, labels=['Unknown'])

def test_augment_data():
    a = [[1, 2, 3], [4, 5, 6]]
    x = np.array(a)
    a = ['label1', 'label2', 'label3']
    y = np.array(a)
    model._augment_data(x, y)


def test_get_features():
    with open('tests/test.pcap', 'a'):
        os.utime('tests/test.pcap', None)
    model.get_features('tests/test.pcap')


def test_save_and_load():
    load_path = 'DeviceClassifier/OneLayer/models/OneLayerModel.pkl'
    save_path = 'tests/test_model'
    model2 = Model(duration=300)
    model2.load(load_path, jsn=False)
    model2.save(save_path, jsn=True)
    os.path.isfile(save_path)
    model2.load(save_path, jsn=True) is not None
