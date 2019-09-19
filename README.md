# jupyanno [![Build Status](https://travis-ci.org/chestrays/jupyanno.svg?branch=master)](https://travis-ci.org/chestrays/jupyanno) [![codecov](https://codecov.io/gh/chestrays/jupyanno/branch/master/graph/badge.svg)](https://codecov.io/gh/chestrays/jupyanno)

![Preview](preview.gif)

# Overview
The focus of the project is to make acquiring more training data as easy as possible for data-scientists. The project uses Jupyter notebooks and widgets to allow for custom GUI tools to be made and deployed with JupyterHub / BinderHub. The tools are also designed to make tracker user interaction easy as well (how long did a user look, what did they change / zoom into / click). The toolkit is made to be easily extensible and suggestions / improvements are welcome. 

# Tasks Addressed

## Multiple Choice
## Binary Choice (with unknown option)
## Bounding Box (in progress)


# Example Tasks
## Chest X-Ray for Pneumonia Task

Here we have a demo chest X-Ray task where the annotation and viewing of a sample dataset can be done entirely with these tools.

[![Binder Annotation Tool](https://img.shields.io/badge/launch-annotation_tool-red.svg)](https://mybinder.org/v2/gh/chestrays/jupyanno/master?urlpath=%2Fvoila%2Frender%2Fnotebooks%2Fanno_app.ipynb%3Fuser%3Drandom_githubber)
[![Binder Dashboard Tool](https://img.shields.io/badge/launch-dashboard-blue.svg)](https://mybinder.org/v2/gh/chestrays/jupyanno/master?urlpath=%2Fvoila%2Frender%2Fnotebooks%2Fdashboard.ipynb%3Fuser%3Drandom_githubber)
[![Google Sheets Results](https://img.shields.io/badge/show-sheets-green.svg)](https://docs.google.com/spreadsheets/d/1T02tRhe3IUUHYsMchc7hmH8nVI3uR0GffdX1PNxKIZA/edit#gid=1178875150)

[![Pneumonia Sheets Results](https://img.shields.io/badge/show-pneumonia_sheets-green.svg)](https://docs.google.com/spreadsheets/d/1JUCLX_17JIGit0Nk4wphgTHlmji9u9PYPmyf_9Wscvg/edit#gid=1062358074)

# Installation
The widgets can be installed by running 
```
pip install git+https://github.com/chestrays/jupyanno/
```

For a more complete environment you can use [repo2docker](https://github.com/jupyter/repo2docker) to create a Docker Image from the repository
