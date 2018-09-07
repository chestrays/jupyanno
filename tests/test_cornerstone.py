import base64
import os

import numpy as np

from jupyanno.cornerstone import encode_numpy_b64


def test_encoding():
    """
    Test the encoding and decoding using the test image from cornerstone
    :return:
    """
    with open(os.path.join('tests',
                           'cs_test_img.txt'), 'r') as f:
        b64_data = f.read().strip()
        k = base64.b64decode(b64_data)
        kk = np.frombuffer(k, dtype=np.uint16).reshape((256, 256))
        encoded_kk = encode_numpy_b64(kk)
    assert all([a == b
                for a, b in
                zip(encoded_kk, b64_data)]), 'Encoded arrays should match'
