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
test_dir = 'tests'


def test_path_img():
    img_dir = mkdtemp()
    im_obj = Image.fromarray(np.eye(3, dtype=np.uint8))
    out_path = os.path.join(img_dir, 'test.png')
    im_obj.save(out_path)

    enc_img_str = utils.path_to_img(out_path)
    assert enc_img_str[:40] == should_enc_begin, 'Image not encoded correctly!'


def test_dcm_loader():
    out_img = utils.load_image_multiformat(os.path.join(test_dir,
                                                        'test_lung_ct.dcm'))
    assert out_img.shape == (512, 512), 'Shape is correct'
    assert out_img.min() == 0, 'Min is correct'
    assert out_img.max() == 2449, 'Max is correct'
    assert out_img.dtype == np.uint16, 'Type is correct'

    out_img = utils.load_image_multiformat(os.path.join(test_dir,
                                                        'test_lung_ct.dcm'),
                                           normalize=True)
    assert out_img.shape == (512, 512), 'Shape is correct'
    assert out_img.min() == 0, 'Min is correct'
    assert out_img.max() == 255, 'Max is correct'
    assert out_img.dtype == np.uint8, 'Type is correct'

    out_pil = utils.load_image_multiformat(os.path.join(test_dir,
                                                        'test_lung_ct.dcm'),
                                           as_pil=True)

    assert out_pil.info == {}, 'Shape is correct'
    assert out_pil.mode == 'RGB', 'Mode is correct'


def test_png_loader():
    img_dir = mkdtemp()
    im_obj = Image.fromarray(np.eye(3, dtype=np.uint8))
    out_path = os.path.join(img_dir, 'test.png')
    im_obj.save(out_path)

    out_img = utils.load_image_multiformat(out_path)
    assert out_img.shape == (3, 3), 'Shape is correct'
    assert out_img.min() == 0, 'Min is correct'
    assert out_img.max() == 1, 'Max is correct'
    assert out_img.dtype == np.uint8, 'Type is correct'
