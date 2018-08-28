import json
import os
from collections import namedtuple
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

from .utils import safe_json_load

TaskData = namedtuple('TaskData',
                      ['task', 'data_df', 'label_col', 'image_key_col',
                       'base_img_dir', 'base_sheet_url', 'sheet_id'])


def read_task_file(in_path):
    with open(in_path, 'r') as f:
        annotation_task = json.load(f)
        data_df = pd.DataFrame(annotation_task['dataset']['dataframe'])
        label_col = annotation_task['dataset']['output_labels']
        image_key_col = annotation_task['dataset']['image_path']
        base_img_dir = annotation_task['dataset']['base_image_directory']
        base_img_dir = os.path.join(os.path.dirname(in_path), base_img_dir)
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


def read_annotation(raw_df):
    """
    Read and parse annotation stored in dataframe format
    :param raw_df:
    :return:
    >>> test_df = pd.DataFrame({'annotator': ['rad_bob']})
    >>> test_df['Timestamp'] = '2018-10-10'
    >>> test_df['viewing_info'] = '{"viewing_time": 5, "views": [1,2,3]}'
    >>> test_df['label'] = 'No'
    >>> test_df['time'] = "5"
    >>> out_df = read_annotation(test_df)
    Found 1 completed annotations
    >>> out_df['viewing_info_dict'].values[0]
    {'viewing_time': 5, 'views': [1, 2, 3]}
    >>> out_df.iloc[0,:].to_dict()
    {'annotator': 'rad_bob', 'Timestamp': Timestamp('2018-10-10 00:00:00'), 'viewing_info': '{"viewing_time": 5, "views": [1,2,3]}', 'label': 'No', 'time': 5.0, 'viewing_time': 5, 'viewing_info_dict': {'viewing_time': 5, 'views': [1, 2, 3]}, 'annotator_class': 'rad', 'annotator_name': 'bob', 'answer_negativity': True}
    """
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
