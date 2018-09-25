import json
import os
import time
from difflib import SequenceMatcher
from itertools import product
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from ipywidgets.embed import embed_snippet
from jupyanno import widgets
from jupyanno.task import TaskData

test_path = 'tests'


def prep_test_image():
    my_temp_dir = mkdtemp()
    my_temp_img = 'A.png'
    Image.fromarray(np.eye(3, dtype=np.uint8)).save(
        os.path.join(my_temp_dir, my_temp_img)
    )
    return my_temp_dir, my_temp_img


def prep_test_task():
    my_temp_dir, my_temp_img = prep_test_image()

    test_data_df = pd.DataFrame({'MyLabel': ['A'],
                                 'MyImageKey': [my_temp_img],
                                 # this key needs to be manually removed
                                 'path': ['JunkPathArg']
                                 })

    test_task_data = TaskData(
        task={},
        data_df=test_data_df,
        label_col='MyLabel',
        image_key_col='MyImageKey',
        base_img_dir=my_temp_dir,
        base_sheet_url='https://a.b.c/?usp=sharing',
        sheet_id=''
    )
    return test_task_data, my_temp_img


@pytest.mark.parametrize("image_panel_type",
                         widgets.IMAGE_VIEWERS.keys())
def test_binaryclasstask(image_panel_type):
    c_task, image_id = prep_test_task()
    global has_been_submitted

    def dummy_submit_func(mc_ans, **kwargs):
        global has_been_submitted
        has_been_submitted = True
        assert mc_ans.task == 'Who Knows!', 'Task should be correct'
        assert mc_ans.annotation_mode == 'BinaryClass', 'Binary Problem'

    for i in range(5):
        # make sure results are reproducible by running multiple times
        q_dict = {'Nein': 'iie!'}
        bct = widgets.BinaryClassTask(['Ja', 'Nein'],
                                      task_data=c_task,
                                      unknown_option=None,
                                      image_panel_type=image_panel_type,
                                      seed=0,
                                      question_dict=q_dict)
        widget_code = str(bct.get_widget())
        assert widgets.MultipleChoiceQuestion.DEFAULT_PREFIX in widget_code, 'Default Question should be inside'
        assert 'iie!' not in widget_code, 'Question should not be inside'
        assert 'IntProgress(value=0' in widget_code, 'Initial progress is 0'

        view_info = json.loads(bct.get_viewing_info())
        if image_panel_type == 'PlotlyImageViewer':
            assert len(view_info['zoom']) == 1, 'One zoom event'
            assert view_info['zoom'][0]['x'] is None, 'X zoom should be empty'
            assert view_info['zoom'][0]['y'] == [0, 3.0], 'Y should be set'

        has_been_submitted = False
        bct.on_submit(dummy_submit_func)
        assert not has_been_submitted, "Preclicking Submit"
        mc_answer = widgets.MultipleChoiceAnswer('False', 'Who Knows!')
        bct._local_submit(mc_answer)
        assert has_been_submitted, "After Clicking Submit"

        widget_code = str(bct.get_widget())
        assert 'IntProgress(value=1' in widget_code, 'Progress is 1'

        assert bct.current_image_id == image_id, 'Image ID should match'

        view_info = json.loads(bct.get_viewing_info())
        assert view_info['viewing_time'] < 0.5, 'Viewing time should be short'
        assert bct.answer_widget.question == 'Nein', 'Question should be Ja'
        widget_code = str(bct.get_widget())
        assert widgets.MultipleChoiceQuestion.DEFAULT_PREFIX not in widget_code, 'Default Question should not be inside'
        assert 'iie!' in widget_code, 'Question should be inside'
        assert bct.answer_widget.labels == ['Yes',
                                            'No'], "Labels should be set"

    # ensure timing works
    time.sleep(0.5)
    view_info = json.loads(bct.get_viewing_info())
    assert view_info['viewing_time'] > 0.5, 'Viewing time should be longer'


def seq_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("im_widget_cls, im_path",
                         product(widgets.IMAGE_VIEWERS.values(),
                                 [os.path.join(test_path, 'test_png.png'),
                                  os.path.join(test_path, 'test_lung_ct.dcm')])
                         )
def test_image_widgets(im_widget_cls, im_path):
    im_widget = im_widget_cls()
    im_widget.clear_image()
    widget_pre_image = str(im_widget.get_widget())
    im_widget.load_image_path(im_path)
    view_info = json.loads(im_widget.get_viewing_info())
    assert view_info['viewing_time'] < 0.5, 'Viewing time should be short'
    widget_with_image = str(im_widget.get_widget())
    im_widget.clear_image()
    widget_sans_image = str(im_widget.get_widget())

    assert seq_sim(widget_with_image,
                   widget_sans_image) > 0.001, 'Before and after should be similar'
    assert seq_sim(widget_with_image,
                   widget_sans_image) < 0.95, 'Before and after but not too much'
    assert seq_sim(widget_pre_image,
                   widget_sans_image) > 0.95, 'Pre and clear should be similarer'


def test_cornerstone_widget():
    """
    Make sure it complains if we have a color image
    :return:
    """
    cs_widget = widgets.CornerstoneViewer()
    img_path = os.path.join(test_path, 'test_png.png')
    with pytest.warns(UserWarning):
        cs_widget.load_image_path(img_path)


def _embed(wid):
    return embed_snippet(wid.get_widget())


REAL_IMAGE_BHEX = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAg'


def test_plotly_widget():
    pl_nobc_widget = widgets.PlotlyImageViewer(with_bc=False)
    nobc_html = _embed(pl_nobc_widget)
    # assert 'Brightness:' not in nobc_html
    pl_widget = widgets.PlotlyImageViewer(with_bc=True)
    empty_html = _embed(pl_widget)
    assert 'Brightness:' in empty_html
    assert len(empty_html) > len(nobc_html)
    empty_state = pl_widget._g.get_state()
    assert 'images' not in empty_state['_layout'], "Should have no images"

    assert REAL_IMAGE_BHEX not in empty_html, 'No Image'
    img_path = os.path.join(test_path, 'test_png.png')
    pl_widget.load_image_path(img_path)
    loaded_state = pl_widget._g.get_state()
    assert len(loaded_state['_layout']['images']) == 1
    loaded_html = _embed(pl_widget)

    assert REAL_IMAGE_BHEX in loaded_html, 'Should have image'
    assert len(loaded_html) - len(
        empty_html) > 100000, 'code should be much bigger'
