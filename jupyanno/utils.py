import base64
import base64 as b64
import inspect
import json
from io import BytesIO

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, Javascript
from PIL import Image
from six.moves.urllib.parse import urlparse, parse_qs


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


def get_app_user_id():
    """
    Get the userid from the `jupyter_notebook_url`
    injected by the appmode extension (if in use)
    otherwise return a 'nobody'
    :return: appmode username of current user
    >>> get_app_user_id()
    'nobody'
    >>> jupyter_notebook_url = 'https://a.b.c?user=dan#hello'
    >>> get_app_user_id()
    'dan'
    """
    cur_url = globals().get('jupyter_notebook_url', None)
    if cur_url is None:
        # black magic to get the 'injected' variable
        frame = inspect.currentframe()
        try:
            out_locals = frame.f_back.f_locals
            cur_url = out_locals.get('jupyter_notebook_url', 'nobody')
        finally:
            del frame
    qs_info = parse_qs(urlparse(cur_url).query)
    return qs_info.get('user', ['nobody'])[0]


def safe_json_load(in_str):
    """
    safely load json strings as dictionaries
    :param in_str: string with encoded json
    :return: dict
    >>> safe_json_load('{"bob": 5}')
    {'bob': 5}
    >>> safe_json_load('{"bob": [1,2,3]}')
    {'bob': [1, 2, 3]}
    >>> safe_json_load('{"bob": [1,2,3}')
    Invalid json row Expecting ',' delimiter: line 1 column 15 (char 14)
    {}
    """
    try:
        return json.loads(in_str)
    except Exception as e:
        print('Invalid json row {}'.format(e))
        return dict()


def _wrap_uri(data_uri):
    return "data:image/png;base64,{0}".format(data_uri)


def raw_html_render(temp_df):
    """
    For rendering html tables which contain HTML information and
    shouldn't be escaped or cropped
    :param temp_df:
    :return:
    >>> f=pd.DataFrame({'k': [1], 'b': ['<a>']})
    >>> print(f.to_html())
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>k</th>
          <th>b</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>&lt;a&gt;</td>
        </tr>
      </tbody>
    </table>
    >>> print(raw_html_render(f))
    <table border="1" class="dataframe table table-striped table-hover">
      <thead>
        <tr style="text-align: right;">
          <th>k</th>
          <th>b</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td><a></td>
        </tr>
      </tbody>
    </table>
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


def fancy_format(in_str, **kwargs):
    """
    A format command for strings that already have lots of curly brackets (ha!)
    :param in_str:
    :param kwargs: the arguments to substitute
    :return:
    >>> fancy_format('{haha} {dan} {bob}', dan = 1, bob = 2)
    '{haha} 1 2'
    """
    new_str = in_str.replace('{', '{{').replace('}', '}}')
    for key in kwargs.keys():
        new_str = new_str.replace('{{%s}}' % key, '{%s}' % key)
    return new_str.format(**kwargs)


def encode_numpy_b64(in_img):
    # type: (np.ndarray) -> str
    """
    Encode numpy arrays as b64 strings
    :param in_img:
    :return:
    >>> encode_numpy_b64(np.eye(2))
    'AAAAAAAA8D8AAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D8='
    """
    return b64.b64encode(in_img.tobytes()).decode()
