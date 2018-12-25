from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import sys
print(sys.version)
import keras
print(keras.__version__)
import tensorflow
print(tensorflow.__version__)
