"""Code for reading and writing results to google sheets"""
from bs4 import BeautifulSoup
import requests
import warnings
import json
import pandas as pd
from six.moves.urllib.parse import urlparse, parse_qs
from six.moves.urllib.request import urlopen

_CELLSET_ID = "AIzaSyC8Zo-9EbXgHfqNzDxVb_YS_IIZBWtvoJ4"


def get_task_sheet(in_task):
    return get_sheet_as_df(sheet_api_url(in_task.sheet_id), _CELLSET_ID)


def get_sheet_as_df(base_url, kk, columns="A:AG"):
    """
    Gets the sheet as a list of Dicts (directly importable to Pandas)
    :return:
    """
    try:
        # TODO: we should probably get the whole sheet
        all_vals = "{base_url}/{cols}?key={kk}".format(base_url=base_url,
                                                       cols=columns,

                                                       kk=kk)
        t_data = json.loads(urlopen(all_vals).read().decode('latin1'))[
            'values']
        frow = t_data.pop(0)

        return pd.DataFrame([
            dict([(key, '' if idx >= len(irow) else irow[idx])
                  for idx, key in enumerate(frow)]) for irow in
            t_data])
    except IOError as e:
        warnings.warn(
            'Sheet could not be accessed, check internet connectivity, \
            proxies and permissions: {}'.format(
                e))
        return pd.DataFrame([{}])


def sheet_api_url(sheet_id):
    return "https://sheets.googleapis.com/v4/spreadsheets/{id}/values".format(
        id=sheet_id)


def get_questions(in_url):
    res = urlopen(in_url)
    soup = BeautifulSoup(res.read(), 'html.parser')

    def get_names(f):
        return [v for k, v in f.attrs.items() if 'label' in k]

    def get_name(f):
        return get_names(f)[0] if len(
            get_names(f)) > 0 else 'unknown'

    all_questions = soup.form.findChildren(
        attrs={'name': lambda x: x and x.startswith('entry.')})
    return {get_name(q): q['name'] for q in all_questions}


def submit_response(form_url, cur_questions, verbose=False, **answers):
    submit_url = form_url.replace('/viewform', '/formResponse')
    form_data = {'draftResponse': [],
                 'pageHistory': 0}
    for v in cur_questions.values():
        form_data[v] = ''
    for k, v in answers.items():
        if k in cur_questions:
            form_data[cur_questions[k]] = v
        else:
            warnings.warn('Unknown Question: {}'.format(k), RuntimeWarning)
    if verbose:
        print(form_data)
    user_agent = {'Referer': form_url,
                  'User-Agent': "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537\
                  .36 (KHTML, like Gecko) Chrome/28.0.1500.52 Safari/537.36"}
    return requests.post(submit_url, data=form_data, headers=user_agent)
