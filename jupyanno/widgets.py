"""The widgets are the heart of jupyanno package and bring together
tasks as task panels (with images) and answer panels (with buttons
and multiple choice optiosn)"""
import os
import ipywidgets as ipw
from glob import glob
import json
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from time import time
from collections import namedtuple, defaultdict

from plotly.widgets import GraphWidget
import plotly.graph_objs as go
import plotly.plotly as py
import base64
from PIL import ImageEnhance as ie

MultipleChoiceAnswer = namedtuple(
    'MultipleChoiceAnswer', ['answer', 'question'])
TaskResult = namedtuple(
    'TaskResult',
    ['annotation_mode', 'task', 'item_id', 'label', 'viewing_info', 'comment'])

VIEWER_WIDTH = '600px'

# TODO: this needs to be moved to package resources
if os.path.exists('load.gif'):
    with open('load.gif', 'rb') as f:
        LOAD_ANIMATION = f.read()
else:
    LOAD_ANIMATION = b''


class WidgetObject:
    """class to make non-widgets seem more widgety"""

    def __init__(self, widget_obj):
        self._widget_obj = widget_obj

    def get_widget(self):
        return self._widget_obj

    def _ipython_display_(self):
        display(self.get_widget())


class SimpleImageViewer(WidgetObject):
    def __init__(self):
        self.cur_image_view = ipw.Image(layout=ipw.Layout(width=VIEWER_WIDTH),
                                        disabled=True)
        self.loaded_time = None
        super().__init__(self.cur_image_view)

    def clear_image(self):
        self.cur_image_view.format = 'gif'
        self.cur_image_view.value = LOAD_ANIMATION

    def load_image_path(self, path):
        c_img = Image.open(path)
        bio_obj = BytesIO()
        c_img.save(bio_obj, format='png')
        bio_obj.seek(0)
        self.cur_image_view.format = 'png'
        self.cur_image_view.value = bio_obj.read()

        self.loaded_time = time()

    def get_viewing_info(self):
        out_info = {}
        if self.loaded_time is not None:
            out_info['viewing_time'] = time() - self.loaded_time
        return json.dumps(out_info)


def wrap_bytes_as_uri(in_byte_obj):
    b64_str = base64.b64encode(in_byte_obj.read()).decode(
        "ascii").replace("\n", "")
    return "data:image/png;base64,{0}".format(b64_str)


def image_dict(c_img):
    bio_obj = BytesIO()
    c_img.save(bio_obj, format='png')
    bio_obj.seek(0)
    nice_uri = wrap_bytes_as_uri(bio_obj)
    return dict(source=nice_uri,
                x=0,
                sizex=c_img.width,
                y=c_img.height,
                sizey=c_img.height,
                xref="x",
                yref="y",
                opacity=1.0,
                sizing="stretch",
                layer="below"
                )


class PlotlyImageViewer(WidgetObject):
    def __init__(self, width=VIEWER_WIDTH, with_bc=True):
        self._g = go.FigureWidget(data=[{
            'x': [0, 1],
            'y': [0, 1],
            'mode': 'markers',
            'marker': {'opacity': 0}}])
        with self._g.batch_update():
            self._g.layout.xaxis.visible = False
            self._g.layout.yaxis.visible = False
            # this can constract the axis (but we leave it off for now since it messes up some images)
            self._g.layout.yaxis.scaleanchor = 'x'
            self._g.layout.margin = {'l': 0, 'r': 0, 't': 0, 'b': 0}

        self.scale_factor = 1.0

        self._brightness = ipw.FloatSlider(value=1.0,
                                           min=0, max=3.5,
                                           description='Brightness:',
                                           continuous_update=False)
        self._contrast = ipw.FloatSlider(value=1.0,
                                         min=0, max=3.5,
                                         description='Contrast:',
                                         continuous_update=False)

        def _select_to_brightness(_1, _2, ls_data):
            def _update_value(in_obj, x_vals):
                diff = (x_vals[0] - x_vals[-1]) / 512
                max_val = in_obj.max
                min_val = in_obj.min
                in_obj.value = np.clip(in_obj.value + diff, min_val, max_val)

            _update_value(self._brightness, ls_data.xs)
            _update_value(self._contrast, ls_data.ys)
            if not with_bc:
                self._update_image(False)

        self._g.data[0].on_selection(_select_to_brightness)

        self._plot_title = ipw.Label('')

        self.clear_image()

        self._g.layout.on_change(
            lambda *args: self._handle_zoom(*args), 'xaxis.range',
            'yaxis.range')
        self.loaded_time = None
        if with_bc:
            self._brightness.observe(
                lambda x: self._update_image(False), names='value')
            self._contrast.observe(
                lambda x: self._update_image(False), names='value')
            bc_tools = [ipw.HBox([self._brightness, self._contrast])]
        else:
            bc_tools = []

        bc_tools += [self._plot_title]
        super().__init__(
            ipw.VBox(bc_tools + [self._g],
                     layout=ipw.Layout(width=width, height="768px"))
        )

    def clear_image(self):
        self._raw_img = None
        self._contrast.value = 1.0
        self._brightness.value = 1.0
        self.out_info = defaultdict(list)
        with self._g.batch_update():
            self._g.layout.images = []
            self._g.layout.title = 'Loading...'
            self._plot_title.value = 'Loading...'
            self._g.layout.dragmode = 'zoom'  # or can be set to pan
            self._g.layout.hovermode = False  # get ride of the annoying popup in the corners

    def load_image_path(self, in_path, **kwargs):
        self.clear_image()
        self._raw_img = Image.open(in_path)
        title = ''
        title_args = kwargs.copy()
        min_args = ['View Position', 'Patient Age', 'Patient Gender']
        for c_arg in min_args:
            if c_arg not in title_args:
                title_args[c_arg] = ''

        title = 'Patient: {Patient Age}{Patient Gender}, View Position: {View Position}'.format(
            **title_args)
        self._update_image(True, title=title)

    def _update_image(self, refresh_view=True, title=''):
        if self._raw_img is not None:
            cont_img = ie.Contrast(self._raw_img).enhance(self._contrast.value)
            proc_img = ie.Brightness(cont_img).enhance(self._brightness.value)
            img_dict = image_dict(proc_img)
            if refresh_view:
                with self._g.batch_update():
                    self._g.data[0].x = [0,
                                         img_dict['sizex'] * self.scale_factor]
                    self._g.data[0].y = [0,
                                         img_dict['sizey'] * self.scale_factor]
                    self._g.layout.images = [img_dict]
                    self._plot_title.value = title
                    self._g.layout.yaxis.range = [0, img_dict[
                        'sizey'] * self.scale_factor]

                self.loaded_time = time()
            else:
                self._g.layout.images = [img_dict]

    def _handle_zoom(self, layout, xrange, yrange):
        self.out_info['zoom'].append({'x': xrange, 'y': yrange})

    def get_viewing_info(self):
        if self.loaded_time is not None:
            self.out_info['viewing_time'] = time() - self.loaded_time
        return json.dumps(self.out_info)


class MultipleChoiceQuestion(WidgetObject):
    def __init__(self, question, labels, question_prefix='', width="150px",
                 buttons_per_row=3):
        self.question_box = ipw.HTML(value='')
        self.labels = labels
        self.width = width
        self.buttons_per_row = buttons_per_row
        self._make_buttons(labels)

        self.question_prefix = question_prefix
        self.set_question(question)

        self.submit_func = None
        super().__init__(ipw.VBox([self.question_box] + self.button_rows,
                                  layout=ipw.Layout(width="250px")))

    def disable_buttons(self, disabled_status=True):
        for c_button in self._button_objs:
            c_button.disabled = disabled_status

    def on_submit(self, submit_func):
        self.submit_func = submit_func

    def set_question(self, question):
        self.question = question
        self.question_box.value = '<h2>{} {}?</h2>'.format(
            self.question_prefix, self.question)
        self.disable_buttons(False)

    def mk_btn(self, description):
        btn = ipw.Button(description=description,
                         layout=ipw.Layout(width=self.width))
        if description == 'Yes':
            btn.style.button_color = 'lightgreen'
        elif description == 'No':
            btn.style.button_color = 'pink'

        def on_click(btn):
            if btn is not None and self.submit_func is not None:
                self.disable_buttons()
                self.submit_func(MultipleChoiceAnswer(
                    answer=btn.description, question=self.question))

        btn.on_click(on_click)
        return btn

    def _make_buttons(self, button_ids):
        self.button_rows = []
        self._button_objs = []
        c_row = []
        for i, but_name in enumerate(button_ids, 1):
            c_button = self.mk_btn(but_name)
            c_row += [c_button]
            self._button_objs.append(c_button)
            if (i % self.buttons_per_row) == 0:
                self.button_rows += [ipw.HBox(c_row)]
                c_row = []
        self.button_rows += [ipw.HBox(c_row)]


class AbstractClassificationTask(WidgetObject):
    def __init__(self, labels, task_data, seed=None, maximum_count=None,
                 with_bc=False,
                 image_panel_width=VIEWER_WIDTH):
        self.labels = labels
        self._image_dict = {
            c_row[task_data.image_key_col]: (
            os.path.join(task_data.base_img_dir,
                         c_row[
                             task_data.image_key_col]),
            c_row)
            for _, c_row in task_data.data_df.iterrows()}
        self._image_df = task_data.data_df

        self.image_keys = sorted(list(self._image_dict.keys()))
        self.submit_event = None
        self.task_widget = PlotlyImageViewer(with_bc=with_bc,
                                             width=image_panel_width)
        self.answer_widget.on_submit(lambda x: self._local_submit(x))
        self.current_image_id = None
        self.set_seed(seed)
        self.maximum_count = maximum_count
        self._comment_field = ipw.Textarea(layout=ipw.Layout(width="200px"))

        self._progress_bar = ipw.IntProgress(
            value=0,
            min=0,
            max=len(images) if maximum_count is None else maximum_count,
            description='Items Done:',
            bar_style='info',
            orientation='horizontal'
        )
        self._local_submit(MultipleChoiceAnswer(question=None, answer=None))

        # we want to append to it later
        self._answer_region = ipw.VBox([self.answer_widget.get_widget()])

        super().__init__(
            ipw.HBox([
                ipw.VBox([self._progress_bar,
                          self.task_widget.get_widget()
                          ], layout=ipw.Layout(width=image_panel_width)
                         ),
                self._answer_region
            ])
        )

    def get_viewing_info(self):
        return self.task_widget.get_viewing_info()

    def set_seed(self, seed):
        if seed is not None:
            np.random.seed(seed)

    def on_submit(self, on_submit):
        self.submit_event = on_submit

    def _submit(self, mc_answer):
        raise NotImplementedError(
            "Subclass needs to have its own implementation of _submit")

    def _update_image(self, image_key):
        # update image
        img_path, img_kwargs = self._image_dict[image_key]
        self.task_widget.load_image_path(img_path, **img_kwargs)
        self.current_image_id = image_key

    def _local_submit(self, mc_answer):
        # harvest results
        generic_info = dict(item_id=self.current_image_id,
                            viewing_info=self.get_viewing_info(),
                            comment=self._comment_field.value)
        # clear the image
        self.task_widget.clear_image()
        if mc_answer.answer is not None:
            self._progress_bar.value += 1

        c_task_info = self._submit(mc_answer)
        # submit results to backend
        c_task = TaskResult(**generic_info,
                            **c_task_info)

        if self.submit_event is not None:
            self.submit_event(c_task)
        if (self.maximum_count is not None) and (
                self._progress_bar.value == (self.maximum_count - 1)):
            self._comment_field.value = 'Comments or Feedback?'
            self._comment_field.rows = 8
            self._answer_region.children = (
                                               self._comment_field,) + self._answer_region.children
        if (self.maximum_count is not None) and (
                self._progress_bar.value >= self.maximum_count):
            self.task_widget.clear_image()
            self.answer_widget.get_widget().close()
            self._comment_field.close()
            self._progress_bar.bar_style = 'success'


class MultiClassTask(AbstractClassificationTask):
    def __init__(self, labels, task_data, seed=None, max_count=None):
        self.answer_widget = MultipleChoiceQuestion(
            'Select the most appropriate label for the given image', labels)
        super().__init__(labels, task_data, seed, max_count)

    def _submit(self, mc_answer):
        c_task_dict = dict(annotation_mode='MultiClass',
                           task=','.join(self.labels),
                           label=mc_answer.answer
                           )

        # get next question
        image_key = np.random.choice(self.image_keys)
        self._update_image(image_key)
        return c_task_dict


class BinaryClassTask(AbstractClassificationTask):
    def __init__(self, labels, task_data, unknown_option, seed=None,
                 max_count=None):
        answer_choices = ['Yes', 'No']
        if unknown_option is not None:
            answer_choices.append(unknown_option)
        self.answer_widget = MultipleChoiceQuestion('',
                                                    answer_choices,
                                                    question_prefix='Does this patient have',
                                                    buttons_per_row=1)
        super().__init__(labels, task_data, seed, max_count)

    def _submit(self, mc_answer):
        c_task_dict = dict(annotation_mode='BinaryClass',
                           task=mc_answer.question,
                           label=mc_answer.answer)

        # get next question
        image_key = np.random.choice(self.image_keys)
        question = np.random.choice(self.labels)
        # update image
        self._update_image(image_key)
        self.answer_widget.set_question(question)
        return c_task_dict
