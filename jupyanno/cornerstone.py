"""tools for making self-contained cornerstone widgetsx"""

import json
from typing import Callable, Tuple

import numpy as np
from IPython.display import Javascript, display

from .utils import fancy_format, encode_numpy_b64

"""
useful tests to implement
with open('img_junk.txt', 'r') as f:
    b64_data = f.read().strip()
    k = b64.b64decode(b64_data)
    kk = np.frombuffer(k, dtype=np.uint16).reshape((256, 256))
assert sum([a==b for a, b in zip(encode_img(kk), b64_data)])==len(b64_data))
"""


def gen_numpy_panel(panel_text, panel_arr,
                    as_full_page=True, panel_id='panel1'):
    # type: (str, np.ndarray, bool, str) -> Tuple[str, Callable, str]
    """
    Create a full numpy panel
    :param panel_text: the name to put on the panel
    :param panel_arr: the image to show in cornerstone
    :param as_full_page: if the <html> and body tags should be generated or if
        it is meant to run inside of ipython
    :param panel_id: the component name of the div
    :return: string or string and functions
    >>> html, _, _ = gen_numpy_panel('test', np.eye(3))
    >>> len(html)
    9200
    >>> html.split('intercept')[-1][:5]
    ': 0.0'
    >>> html.split('var all_images_list')[-1][:19]
    ' = [["numpy://0"]];'
    """
    numpy_load_js = make_pyloader_js(panel_arr)
    panel_code = generate_panels(panel_names=[panel_id],
                                 panel_ids=[panel_id],
                                 panel_texts=[panel_text])

    mp_script_js = fancy_format(multipanel_script_js,
                                PANEL_NAMES=json.dumps([panel_id]),
                                PANEL_IDS=json.dumps([panel_id]),
                                PATHS_LIST=json.dumps([['numpy://0']]),
                                ADDITIONAL_JS_CODE=numpy_load_js,
                                )

    body_html = fancy_format(cornerstone_body_html, PANEL_HTML=panel_code)
    if as_full_page:
        return fancy_format(cornerstone_core_multipanel_html,
                            JS_IMPORTS=js_html,
                            CSS_CODE=css_html,
                            BODY_HTML=body_html,
                            JS_CODE=mp_script_js), lambda: None, mp_script_js
    else:
        def prep_func():
            for c_js in js_names:
                display(Javascript(url=c_js))
    return body_html, prep_func, mp_script_js


def make_pyloader_js(in_arr):
    # type: (np.ndarray) -> str
    """
    Makes an image loader from a numpy array
    :param in_arr: numpy array
    :return:
    >>> simple_html = make_pyloader_js(np.eye(3))
    >>> len(simple_html)
    1799
    >>> 'loadPythonImage' in simple_html
    True
    >>> simple_html.split('sizeInBytes')[-1][:11]
    ': 3 * 3 * 2'
    >>> ct_img = 6000*np.eye(7)-3000
    >>> simple_html = make_pyloader_js(ct_img)
    >>> simple_html.split('sizeInBytes')[-1][:11]
    ': 7 * 7 * 2'
    >>> simple_html.split('intercept')[-1][:8]
    ': -3000.'
    >>> simple_html.split('minPixelValue')[-1][:8]
    ': -3000.'
    """
    if len(in_arr.shape) != 2:
        raise NotImplementedError(
            'Does not handle shape:{}'.format(in_arr.shape))
    offset = in_arr.min()
    min_val = offset
    max_val = in_arr.max()
    wc = in_arr.mean()
    ww = in_arr.std()
    new_arr = in_arr.astype(np.float64) - offset
    new_arr = new_arr.astype(np.uint16)
    out_b64 = encode_numpy_b64(new_arr)
    h, w = in_arr.shape
    return fancy_format(PYLOAD_JS, window_center=wc, window_width=ww, width=w,
                        height=h, image_data=out_b64,
                        offset=offset, min_val=min_val, max_val=max_val)


def make_pyimage_js(in_arr):
    # type: (np.ndarray) -> str
    """
    Makes a Cornerstone image from a numpy array
    :param in_arr: numpy array
    :return:
    >>> simple_html = make_pyimage_js(np.eye(3))
    >>> len(simple_html)
    1103
    >>> 'loadPythonImage' in simple_html
    False
    >>> simple_html.split('sizeInBytes')[-1][:11]
    ': 3 * 3 * 2'
    >>> ct_img = 6000*np.eye(7)-3000
    >>> simple_html = make_pyimage_js(ct_img)
    >>> simple_html.split('sizeInBytes')[-1][:11]
    ': 7 * 7 * 2'
    >>> simple_html.split('intercept')[-1][:8]
    ': -3000.'
    >>> simple_html.split('minPixelValue')[-1][:8]
    ': -3000.'
    """
    if len(in_arr.shape) != 2:
        raise NotImplementedError(
            'Does not handle shape:{}'.format(in_arr.shape))
    offset = in_arr.min()
    min_val = offset
    max_val = in_arr.max()
    wc = in_arr.mean()
    ww = in_arr.std()
    new_arr = in_arr.astype(np.float64) - offset
    new_arr = new_arr.astype(np.uint16)
    out_b64 = encode_numpy_b64(new_arr)
    h, w = in_arr.shape
    return fancy_format(PYIMG_JS, window_center=wc, window_width=ww, width=w,
                        height=h, image_data=out_b64,
                        offset=offset, min_val=min_val, max_val=max_val)


PYIMG_JS = """
function str2ab(str) {
    var buf = new ArrayBuffer(str.length*2); // 2 bytes for each char
    var bufView = new Uint16Array(buf);
    var index = 0;
    for (var i=0, strLen=str.length; i<strLen; i+=2) {
        var lower = str.charCodeAt(i);
        var upper = str.charCodeAt(i+1);
        bufView[index] = lower + (upper <<8);
        index++;
    }
    return bufView;
}

function parsePixelData(base64PixelData)
{
    var pixelDataAsString = window.atob(base64PixelData);
    var pixelData = str2ab(pixelDataAsString);
    return pixelData;
}
var imageB64Data = "{image_data}"
var imagePixelData = parsePixelData(imageB64Data);
function getPixelData() {
    return imagePixelData;
}
var image = {
    imageId: 'test',
    minPixelValue: {min_val},
    maxPixelValue: {max_val},
    slope: 1.0,
    intercept: {offset},
    windowCenter : {window_center},
    windowWidth : {window_width},
    getPixelData: getPixelData,
    rows: {height},
    columns: {width},
    height: {height},
    width: {width},
    color: false,
    columnPixelSpacing: .8984375,
    rowPixelSpacing: .8984375,
    sizeInBytes: {width} * {height} * 2
};
"""

PYLOAD_JS = """
(function (cs) {
    "use strict";
    function str2ab(str) {
        var buf = new ArrayBuffer(str.length*2); // 2 bytes for each char
        var bufView = new Uint16Array(buf);
        var index = 0;
        for (var i=0, strLen=str.length; i<strLen; i+=2) {
            var lower = str.charCodeAt(i);
            var upper = str.charCodeAt(i+1);
            bufView[index] = lower + (upper <<8);
            index++;
        }
        return bufView;
    }

    function parsePixelData(base64PixelData)
    {
        var pixelDataAsString = window.atob(base64PixelData);
        var pixelData = str2ab(pixelDataAsString);
        return pixelData;
    }
    var imageB64Data = "{image_data}"
    var imagePixelData = parsePixelData(imageB64Data);

    function loadPythonImage(imageId) {
        console.log('Reading Image:'+imageId);
        function getPixelData() {
            return imagePixelData;
        }
        var image = {
            imageId: imageId,
            minPixelValue: {min_val},
            maxPixelValue: {max_val},
            slope: 1.0,
            intercept: {offset},
            windowCenter : {window_center},
            windowWidth : {window_width},
            getPixelData: getPixelData,
            rows: {height},
            columns: {width},
            height: {height},
            width: {width},
            color: false,
            columnPixelSpacing: .8984375,
            rowPixelSpacing: .8984375,
            sizeInBytes: {width} * {height} * 2
        };
        var out_promise=new Promise((resolve) => {
              resolve(image);
            });
        return {
            promise: out_promise,
            cancelFn: undefined
        };
    }
    // register our imageLoader plugin with cornerstone
    cs.registerImageLoader('numpy', loadPythonImage);
}(cornerstone));
"""

base_url = 'https://rawgit.com/kmader/fd00cbfdc172dee9a575924819e05ef7/raw/ee01204f3bb074937feb720865da97c3aeeed828/'

js_names = ["https://unpkg.com/jquery@3.3.1/dist/jquery.js",
            "https://unpkg.com/cornerstone-core@2.2.4/dist/cornerstone.min.js",
            "https://unpkg.com/cornerstone-math@0.1.6/dist/cornerstoneMath.min.js",
            "https://unpkg.com/cornerstone-tools@2.3.9/dist/cornerstoneTools.min.js"]

js_html = '\n'.join(
    [
        """<script type="text/javascript" src="{c_url}"></script>""".format(
            base_url=base_url, c_url=c_url)
        for c_url in js_names])

css_names = ['tachyons.min.css', 'main.css']
css_html = '\n'.join([
    """<link rel="stylesheet" href="{base_url}{c_url}">""".format(
        base_url=base_url, c_url=c_url)
    for c_url in css_names])

panel_code_html = """
<div class="fl w-{width_tag}">
  <div class="dib w-100"><p class="dib pr3 fl">{image_text}</p></div>
  <div id="{image_id}" class="ba ma1" style="width:512px;height:512px;"></div>
  <div><p class="dib pr3 fl" id="{text_id}"></p></div>
</div>
"""


def generate_panels(panel_names, panel_ids, panel_texts=None):
    if panel_texts is None:
        panel_texts = panel_names
    return '\n'.join([fancy_format(panel_code_html,
                                   width_tag=TACHYON_WIDTH_TAGS.get(
                                       len(panel_names), '10'),
                                   image_id=c_name,
                                   image_text=c_text,
                                   text_id=c_id)
                      for c_name, c_id, c_text in
                      zip(panel_names, panel_ids, panel_texts)])


TACHYON_WIDTH_TAGS = {1: '100', 2: '50', 3: 'third', 4: '25', 5: '20',
                      6: '10'}  # tag to use as a function of the number of panels


def make_panel_js(panel_names, panel_ids, panel_paths, first_region):
    return fancy_format(multipanel_script_js,
                        PANEL_NAMES=json.dumps(panel_names),
                        PANEL_IDS=json.dumps(panel_ids),
                        PATHS_LIST=json.dumps(panel_paths),
                        FIRST_REGION=json.dumps(first_region),
                        ADDITIONAL_JS_CODE='',
                        )


cornerstone_body_html = """
  <div class="cf bt b--moon-gray-10">
    <div class="fl w-100 tc">
      <div class="fl w-100">
        <p class="pa3 ph5-ns dib fl">Tools:</p>
        <ul id="tools-list" class='list pa3 ph5-ns lh-copy'>
            <li class="dib ph3 fl"><a id="wwwc" href="#" class="link moon-gray list-group-item">|WW/WC|</a></li>
            <li class="dib ph2 fl"><a id="pan" href="#" class="link moon-gray list-group-item">|Pan|</a></li>
            <li class="dib ph2 fl"><a id="zoom" href="#" class="link moon-gray list-group-item">|Zoom|</a></li>
            <li class="dib ph2 fl"><a id="reset" href="#" class="link moon-gray list-group-item">|Reset|</a></li>
        </ul>
      </div>
      <div class="pa3 ph5-ns fl w-100 tc">
        {PANEL_HTML}
      </div>
    </div>
  </div>
"""

cornerstone_core_multipanel_html = """
<!DOCTYPE html>
<html lang="en" class="moon-gray bg-near-black">
<head>
  <title>CXR Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {CSS_CODE}
</head>
<body class="w-100 sans-serif">
    {BODY_HTML}
    {JS_IMPORTS}
    <script type="text/javascript">{JS_CODE}</script>
</body>

</html>
"""

multipanel_script_js = """
// global variables
var panel_names = {PANEL_NAMES};
var panel_ids = {PANEL_IDS};
var all_images_list = {PATHS_LIST};
var image_panels = new Array(panel_names.length);
var body = null;
var bb_stack = 0;

var serializedState = null
var initialAppState = null;
// disable the default context menu which is there by default
function disableContextMenu(e) {
    $(e).on('contextmenu', function(e) {
        e.preventDefault();
    });
}

function onNewImage(e) {
}

function onImageRendered(e, eventData) {
}

function onInitialState() {
    console.log('captured initial state');
    initialAppState = cornerstoneTools.appState.save(image_panels);
}

function load_dicom_stack(imageIds, element) {
    cornerstone.enable(element);
    var stack = {
        currentImageIdIndex: 0,
        imageIds: imageIds
    };
    console.log('Loading:'+imageIds[stack.currentImageIdIndex]);
    cornerstoneTools.mouseInput.enable(element);
    cornerstoneTools.mouseWheelInput.enable(element);
    return cornerstone.loadImage(imageIds[stack.currentImageIdIndex])
        .then(function (image) {
            var viewport = cornerstone.getDefaultViewportForImage(element, image);
            cornerstone.displayImage(element, image, viewport);
            cornerstoneTools.wwwc.activate(element, 1);
        })
        .catch(function (error) {
            console.log("When loading stack:"+imageIds[0]+" in "+element+" error "+error+" occured");
        });
}

function load_all_stacks(all_paths_list) {

    var eval_list = [];
    for(var i = 0; i < image_panels.length; i++) {
        eval_list.push(load_dicom_stack(all_paths_list[i], image_panels[i]));
    }

    Promise.all(eval_list)
        .then(function(values) {
            var synchronizer = new cornerstoneTools.Synchronizer("CornerstoneStackScroll", cornerstoneTools.stackScrollSynchronizer);
            for(var i = 0; i < image_panels.length; i++) {
            synchronizer.add(image_panels[i]);
            $(image_panels[i]).on("CornerstoneNewImage", onImageRendered);
            }
            onInitialState(image_panels[i]);
        })
    .catch(function (error) {
        console.log("When loading stacks error: " + error + " occured");
    });
}


$(function () {
    // post setup code
    console.log("Ready!");
    for(var i = 0; i < panel_names.length; i++) {
        image_panels[i] = $('#'+panel_names[i]).get(0);
        disableContextMenu(image_panels[i]);
        console.log(panel_names[i]+":"+image_panels[i]);
    }

    body = $('body');

    // Add event handlers to zoom the image in and out
    $('#zoomIn').click(function (element) {
        var viewport = cornerstone.getViewport(element);
        viewport.scale += 0.25;
        cornerstone.setViewport(element, viewport);
    });

    $('#zoomOut').click(function (element) {
        var viewport = cornerstone.getViewport(element);
        viewport.scale -= 0.25;
        cornerstone.setViewport(element, viewport);
    });

    $('#pan').click(function () {
        activate('#pan');
        for(var i = 0; i < image_panels.length; i++) {

            disableAllTools(image_panels[i]);
            cornerstoneTools.pan.activate(image_panels[i], 1);
            cornerstoneTools.zoom.activate(image_panels[i], 2);
        }

    });

    $('#wwwc').click(function () {
        activate('#wwwc');
        for(var i = 0; i < image_panels.length; i++) {
            disableAllTools(image_panels[i]);
            cornerstoneTools.wwwc.activate(image_panels[i], 1);
        }

    });


    $('#zoom').click(function () {
        activate('#zoom');
        for(var i = 0; i < image_panels.length; i++) {
            disableAllTools(image_panels[i]);
            cornerstoneTools.zoom.activate(image_panels[i], 1);
        }
    });


    $('#reset').click(function () {
        for(var i = 0; i < image_panels.length; i++) {
            cornerstone.reset(image_panels[i]);
        }
    });


    $('#savestate').click(function () {
        var appState = cornerstoneTools.appState.save(image_panels);
        serializedState = JSON.stringify(appState);
    });

    $('#restorestate').click(function () {
        var appState = cornerstoneTools.appState.save(image_panels);
        jsntxt = appState[imageIdToolState] = { '': '' }
        var appState = JSON.parse(jsntxt)
        cornerstoneTools.appState.restore(appState);
        console.log('State restored');
    });

    function activate(id) {
        $("#tools-list").find("a.light-purple").removeClass("light-purple");
        $(id).addClass('light-purple');
        $(id).addClass('active');
    }

    function disableAllTools(element) {
        // helper function used by the tool button handlers to disable the active tool
        // before making a new tool active
        cornerstoneTools.wwwc.deactivate(element, 1);
        cornerstoneTools.pan.deactivate(element, 2); // 2 is middle mouse button
        cornerstoneTools.zoom.deactivate(element, 4); // 4 is right mouse button
        cornerstoneTools.length.deactivate(element, 1);
        //cornerstoneTools.ellipticalRoi.deactivate(element, 1);
        cornerstoneTools.rectangleRoi.deactivate(element, 1);
        //cornerstoneTools.angle.deactivate(element, 1);
        //cornerstoneTools.highlight.deactivate(element, 1);
        //cornerstoneTools.freehand.deactivate(element, 1);
    }

    {ADDITIONAL_JS_CODE}

    load_all_stacks(all_images_list);
});
"""
