"""
Most of the coverage comes from doctests but catching exceptions and
handling IO is definitely more tricky there
"""
import os
from tempfile import mkdtemp

import numpy as np
from PIL import Image

from jupyanno import utils

should_enc_begin = """<img src="data:image/png;base64,iVBORw0K"""


def test_path_img():
    img_dir = mkdtemp()
    im_obj = Image.fromarray(np.eye(3, dtype=np.uint8))
    out_path = os.path.join(img_dir, 'test.png')
    im_obj.save(out_path)

    enc_img_str = utils.path_to_img(out_path)
    assert enc_img_str[:40] == should_enc_begin, 'Image not encoded correctly!'
