import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def sec_struct_ohe(secs):
    label_encoder_ = LabelEncoder()
    ohe_arrs = [to_categorical(label_encoder_.fit_transform(np.array(list(sec))), num_classes=3) for sec in secs]
    return np.array(ohe_arrs)