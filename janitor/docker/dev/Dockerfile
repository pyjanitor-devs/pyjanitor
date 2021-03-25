FROM python:3.7
FROM jupyter/minimal-notebook
WORKDIR $HOME
RUN python -m pip install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN python -m pip  install -r requirements.txt  --ignore-installed

RUN python -m pip install --upgrade --no-deps --force-reinstall notebook

#
RUN python -m pip install jupyterthemes
RUN python -m pip install --upgrade jupyterthemes
RUN python -m pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable --user
RUN jupyter nbextension enable jupyter-js-widgets/extension
RUN jupyter nbextension enable jupyter-notebook-gist/notebook-extension
RUN jupyter nbextension enable contrib_nbextensions_help_item/main
RUN jupyter nbextension enable autosavetime/main
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable code_font_size/code_font_size
RUN jupyter nbextension enable code_prettify/code_prettify
RUN jupyter nbextension enable collapsible_headings/main
RUN jupyter nbextension enable comment-uncomment/main
RUN jupyter nbextension enable equation-numbering/main
RUN jupyter nbextension enable execute_time/ExecuteTime
RUN jupyter nbextension enable gist_it/main
RUN jupyter nbextension enable hide_input/main
RUN jupyter nbextension enable spellchecker/main
RUN jupyter nbextension enable toc2/main
RUN jupyter nbextension enable toggle_all_line_numbers/main

USER root
RUN apt-get update
RUN  apt-get -y install python-pydot
RUN  apt-get -y install python-pydot-ng
RUN  apt-get -y install graphviz
RUN rm -rf /var/lib/apt/lists/*
