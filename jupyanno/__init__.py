import base64
import json
import os
import warnings
from collections import namedtuple
from glob import glob
from io import BytesIO
from itertools import cycle
from time import time
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, Javascript, HTML
from PIL import Image
from scipy.stats import binom
from six.moves.urllib.parse import urlparse, parse_qs
from six.moves.urllib.request import urlopen

TaskData = namedtuple('TaskData',
                      ['task', 'data_df', 'label_col', 'image_key_col',
                       'base_img_dir', 'base_sheet_url', 'sheet_id'])


def setup_appmode():
    js_str = """$('#appmode-leave').hide()
        // Hides the edit app button.
        $('#appmode-busy').hide()
        // Hides the kernel busy indicator.
        IPython.OutputArea.prototype._should_scroll = function(lines) {
            return false
            // disable scrolling
        }"""
    sns.set_style("whitegrid", {'axes.grid': False})
    display(Javascript(js_str))


def _get_user_id():
    cur_url = globals().get('jupyter_notebook_url', 'nobody')
    qs_info = parse_qs(urlparse(cur_url).query)
    return qs_info.get('user', ['nobody'])[0]




def safe_json_load(in_str):
    try:
        return json.loads(in_str)
    except Exception as e:
        print('Invalid json row {}'.format(e))
        return dict()


def read_annotation(raw_df):
    annot_df = raw_df.copy()
    annot_df['Timestamp'] = pd.to_datetime(annot_df['Timestamp'])
    annot_df['viewing_time'] = annot_df['viewing_info'].map(
        lambda x: safe_json_load(x).get('viewing_time', 0))
    annot_df['viewing_info_dict'] = annot_df['viewing_info'].map(
        lambda x: safe_json_load(x))
    annot_df['annotator_class'] = annot_df['annotator'].map(
        lambda x: x.split('_')[0])
    annot_df['annotator_name'] = annot_df['annotator'].map(
        lambda x: ' '.join(x.split('_')[1:]) if x.find('_') else x)
    annot_df['answer_negativity'] = annot_df['label'].map(
        lambda x: ('No ' in x) or x == 'No')
    annot_df['time'] = annot_df['time'].map(float)
    print('Found', annot_df.shape[0], 'completed annotations')
    return annot_df


def binary_correct(c_row, label_col):
    """
    determine if the row is correct or not
    :param c_row: row from a dataframe (dict-like)
    :param label_col: the colum where the label information resides
    :return: boolean if the value is correct or not
    >>> test_row = {'task': 'Pneumonia', 'value': 'Pneumonia', 'label': 'Pneumonia'}
    >>> binary_correct(test_row, 'value')
    True
    >>> test_row['value'] = 'Influenza'
    >>> binary_correct(test_row, 'value')
    False
    >>> test_row = {'task': 'Influenza', 'value': 'Pneumonia', 'label': None}
    >>> binary_correct(test_row, 'value')
    True
    """
    if c_row['label'] == c_row[label_col]:
        return True
    elif c_row['label'] is None:
        if c_row[label_col] == c_row['task']:
            # definitely wrong
            return False
        else:
            # definitely right
            return True
    else:
        # if the label is positive but not what we picked
        return False

    return None  # we arent sure if it is right or not


def read_task_file(in_path):
    with open(in_path, 'r') as f:
        annotation_task = json.load(f)
        data_df = pd.DataFrame(annotation_task['dataset']['dataframe'])
        label_col = annotation_task['dataset']['output_labels']
        image_key_col = annotation_task['dataset']['image_path']
        base_img_dir = annotation_task['dataset']['base_image_directory']
        base_sheet_url = annotation_task['google_forms']['sheet_url']
        sheet_id = \
            base_sheet_url.strip('?usp=sharing').strip('/edit').split('/')[-1]
        return TaskData(annotation_task, data_df, label_col, image_key_col,
                        base_img_dir, base_sheet_url, sheet_id)


def show_my_result(name_list, correct_list, num_questions=30, ax1=None):
    n_correct = np.arange(num_questions + 1)
    if ax1 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5), dpi=250)
    binom_pmf = binom.pmf(n_correct, num_questions, 0.5)
    binom_cdf = np.cumsum(binom_pmf)
    ax1.bar(n_correct, binom_pmf / np.max(binom_pmf), color='k', alpha=0.5)

    ax1.plot(n_correct, binom_cdf, 'k-', label='Cumulative')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = cycle(prop_cycle.by_key()['color'])
    y_pos_list = np.linspace(0.2, 0.8, len(name_list))
    for y_pos, name, correct, color in zip(y_pos_list, name_list, correct_list,
                                           color_cycle):
        ax1.axvline(correct, label='{}\nTop {:2.1f}%'.format(
            name, 100 * binom_cdf[correct]), color=color)
        ax1.text(correct, y_pos, name)
    ax1.legend()
    return ax1


def _wrap_uri(data_uri): return "data:image/png;base64,{0}".format(data_uri)


def raw_html_render(temp_df):
    """
    For rendering html tables which contain HTML information and
    shouldn't be escaped or cropped
    :param temp_df:
    :return:
    """
    old_wid = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', -1)
    tab_html = temp_df.to_html(classes="table table-striped table-hover",
                               escape=False,
                               float_format=lambda x: '%2.2f' % x,
                               na_rep='',
                               index=False,
                               max_rows=None,
                               max_cols=None)

    pd.set_option('display.max_colwidth', old_wid)
    return tab_html


def path_to_img(in_path):
    c_img_data = Image.open(in_path)
    c_img_data = c_img_data.convert('RGB')
    out_img_data = BytesIO()
    c_img_data.save(out_img_data, format='png')
    out_img_data.seek(0)  # rewind
    uri = _wrap_uri(base64.b64encode(out_img_data.read()
                                     ).decode("ascii").replace("\n", ""))
    return '<img src="{uri}"/>'.format(uri=uri)
