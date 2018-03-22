# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''Functions for downloading and reading MNIST data (deprecated).'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy
import os

dataset_dir = os.environ['DATASET_DIR']

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  '''
  Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  '''
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  '''Convert class labels from scalars to one-hot vectors.'''
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  '''Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  '''
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def read_data_sets(data_type="train"):
    '''
    Parse or download mnist data if train_dir is empty.

    :param: train_dir: The directory storing the mnist data

    :param: data_type: Reading training set or testing set.It can be either "train" or "test"

    :return:

    ```
    (ndarray, ndarray) representing (features, labels)
    features is a 4D unit8 numpy array [index, y, x, depth] representing each pixel valued from 0 to 255.
    labels is 1D unit8 nunpy array representing the label valued from 0 to 9.
    '''


    if data_type == "train":
        train_image_file = dataset_dir+os.sep+TRAIN_IMAGES
        with open(train_image_file, 'rb') as f:
            train_images = extract_images(f)

        train_label_file = dataset_dir+os.sep+TRAIN_LABELS
        with open(train_label_file, 'rb') as f:
            train_labels = extract_labels(f)
        return train_images, train_labels

    else:
        test_image_file = dataset_dir+os.sep+TEST_IMAGES
        with open(test_image_file, 'rb') as f:
            test_images = extract_images(f)

        test_label_file = dataset_dir+os.sep+TEST_LABELS
        with open(test_label_file, 'rb') as f:
            test_labels = extract_labels(f)
        return test_images, test_labels

if __name__ == "__main__":
    train, _ = read_data_sets("train")
    test, _ = read_data_sets("test")
