import json
import os
import time
from itertools import product
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from jupyanno import widgets
from jupyanno.task import TaskData
from difflib import SequenceMatcher

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
                                 'MyImageKey': [my_temp_img]
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


def test_binaryclasstask():
    c_task, image_id = prep_test_task()
    global has_been_submitted

    def dummy_submit_func(mc_ans, **kwargs):
        global has_been_submitted
        has_been_submitted = True
        assert mc_ans.task == 'Who Knows!', 'Task should be correct'
        assert mc_ans.annotation_mode == 'BinaryClass', 'Binary Problem'

    for i in range(5):
        # make sure results are reproducible by running multiple times
        bct = widgets.BinaryClassTask(['Ja', 'Nein'],
                                      task_data=c_task,
                                      unknown_option=None,
                                      seed=2018,
                                      prefix='Tests are annoying!')
        widget_code = str(bct.get_widget())
        assert 'Tests are annoying' in widget_code, 'Prefix should be inside'
        assert 'IntProgress(value=0' in widget_code, 'Initial progress is 0'

        view_info = json.loads(bct.get_viewing_info())
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
        assert bct.answer_widget.question == 'Ja', 'Question should be Ja'
        assert bct.answer_widget.labels == ['Yes', 'No'], "Labels should be set"

    # ensure timing works
    time.sleep(0.5)
    view_info = json.loads(bct.get_viewing_info())
    assert view_info['viewing_time'] > 0.5, 'Viewing time should be longer'

def seq_sim(a,b):
    return SequenceMatcher(None, a, b).ratio()

@pytest.mark.parametrize("im_widget_cls, im_path",
                         product([widgets.SimpleImageViewer,
                                  widgets.PlotlyImageViewer],
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
                   widget_sans_image) > 0.5, 'Before and after should be similar'
    assert seq_sim(widget_with_image,
                   widget_sans_image) < 0.95, 'Before and after but not too much'
    assert seq_sim(widget_pre_image,
                   widget_sans_image) > 0.95, 'Pre and clear should be similarer'
