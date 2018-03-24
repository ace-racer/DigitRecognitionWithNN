import scipy.misc
import os
import functools
import operator
import gzip
import struct
import array
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import configurations

MNIST_TEST_IMAGES_LOCATION = "MNIST-data/t10k-images.idx3-ubyte"
TEST_IMAGES_LOCATION = configurations.TEST_IMAGES_LOCATION

# Code copied from https://github.com/datapythonista/mnist/blob/master/mnist/__init__.py
# All non required dependencies have been removed


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.
    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise ValueError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise ValueError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise ValueError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise ValueError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


def get_test_images():
    images = None
    with open(MNIST_TEST_IMAGES_LOCATION, "rb") as fr:
        images = parse_idx(fr)
    return images

def generate_test_image(id):
    images = get_test_images()
    test_image_location = TEST_IMAGES_LOCATION.format(id)
    print("Image generated here: " + test_image_location)
    img = scipy.misc.toimage(scipy.misc.imresize(images[id,:,:] * -1 + 256, 10.)).save(test_image_location)




# Functions to test
# get_test_image(1)