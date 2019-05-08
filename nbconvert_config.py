import os
from pathlib import Path

# Configuration file for jupyter-nbconvert.

# ------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
# ------------------------------------------------------------------------------

## This is an application.

## The date format used by logging formatters for %(asctime)s
# c.Application.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
# c.Application.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
# c.Application.log_level = 30

# ------------------------------------------------------------------------------
# JupyterApp(Application) configuration
# ------------------------------------------------------------------------------

## Base class for Jupyter applications

## Answer yes to any prompts.
# c.JupyterApp.answer_yes = False

## Full path of a config file.
# c.JupyterApp.config_file = ''

## Specify a config file to load.
# c.JupyterApp.config_file_name = ''

## Generate default config file.
# c.JupyterApp.generate_config = False

# ------------------------------------------------------------------------------
# NbConvertApp(JupyterApp) configuration
# ------------------------------------------------------------------------------

## This application is used to convert notebook files (*.ipynb) to various other
#  formats.
#
#  WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.

## The export format to be used, either one of the built-in formats, or a dotted
#  object name that represents the import path for an `Exporter` class
# c.NbConvertApp.export_format = 'html'

## read a single notebook from stdin.
# c.NbConvertApp.from_stdin = False

## List of notebooks to convert. Wildcards are supported. Filenames passed
#  positionally will be added to the list.
c.NbConvertApp.notebooks = ["examples/notebooks/*.ipynb"]

## overwrite base name use for output files. can only be used when converting one
#  notebook at a time.
# c.NbConvertApp.output_base = ''

## Directory to copy extra files (figures) to. '{notebook_name}' in the string
#  will be converted to notebook basename
# c.NbConvertApp.output_files_dir = '{notebook_name}_files'

## PostProcessor class used to write the results of the conversion
# c.NbConvertApp.postprocessor_class = ''

## Whether to apply a suffix prior to the extension (only relevant when
#  converting to notebook format). The suffix is determined by the exporter, and
#  is usually '.nbconvert'.
# c.NbConvertApp.use_output_suffix = True

## Writer class used to write the  results of the conversion
# c.NbConvertApp.writer_class = 'FilesWriter'

# ------------------------------------------------------------------------------
# NbConvertBase(LoggingConfigurable) configuration
# ------------------------------------------------------------------------------

## Global configurable class for shared config
#
#  Useful for display data priority that might be used by many transformers

## Deprecated default highlight language as of 5.0, please use language_info
#  metadata instead
# c.NbConvertBase.default_language = 'ipython'

## An ordered list of preferred output type, the first encountered will usually
#  be used when converting discarding the others.
# c.NbConvertBase.display_data_priority = ['text/html', 'application/pdf', 'text/latex', 'image/svg+xml', 'image/png', 'image/jpeg',
# 'text/markdown', 'text/plain']

# ------------------------------------------------------------------------------
# Exporter(LoggingConfigurable) configuration
# ------------------------------------------------------------------------------

# Class containing methods that sequentially run a list of preprocessors on a
# NotebookNode object and then return the modified NotebookNode object and
# accompanying resources dict.


# List of preprocessors available by default, by name, namespace,  instance,
# or type.
c.Exporter.default_preprocessors = [
    "nbconvert.preprocessors.ExecutePreprocessor",
    "nbconvert.preprocessors.coalesce_streams",
    "nbconvert.preprocessors.SVG2PDFPreprocessor",
    "nbconvert.preprocessors.CSSHTMLHeaderPreprocessor",
    "nbconvert.preprocessors.LatexPreprocessor",
    "nbconvert.preprocessors.HighlightMagicsPreprocessor",
    "nbconvert.preprocessors.ExtractOutputPreprocessor",
]

## Extension of the file that should be written to disk
# c.Exporter.file_extension = '.txt'

## List of preprocessors, by name or namespace, to enable.
# c.Exporter.preprocessors = []

# ------------------------------------------------------------------------------
# TemplateExporter(Exporter) configuration
# ------------------------------------------------------------------------------

## Exports notebooks into other file formats.  Uses Jinja 2 templating engine to
#  output new formats.  Inherit from this class if you are creating a new
#  template type along with new filters/preprocessors.  If the filters/
#  preprocessors provided by default suffice, there is no need to inherit from
#  this class.  Instead, override the template_file and file_extension traits via
#  a config file.
#
#  Filters available by default for templates:
#
#  - add_anchor - add_prompts - ansi2html - ansi2latex - ascii_only -
#  citation2latex - comment_lines - convert_pandoc - escape_latex -
#  filter_data_type - get_lines - get_metadata - highlight2html - highlight2latex
#  - html2text - indent - ipython2python - json_dumps - markdown2asciidoc -
#  markdown2html - markdown2latex - markdown2rst - path2url - posix_path -
#  prevent_list_blocks - strip_ansi - strip_dollars - strip_files_prefix -
#  wrap_text

## Dictionary of filters, by name and namespace, to add to the Jinja environment.
# c.TemplateExporter.filters = {}

## formats of raw cells to be included in this Exporter's output.
# c.TemplateExporter.raw_mimetypes = []

##
# c.TemplateExporter.template_extension = '.tpl'

## Name of the template file to use
# c.TemplateExporter.template_file = ''

##
# c.TemplateExporter.template_path = ['.']

# ------------------------------------------------------------------------------
# ASCIIDocExporter(TemplateExporter) configuration
# ------------------------------------------------------------------------------

## Exports to an ASCIIDoc document (.asciidoc)

# ------------------------------------------------------------------------------
# HTMLExporter(TemplateExporter) configuration
# ------------------------------------------------------------------------------

## Exports a basic HTML document.  This exporter assists with the export of HTML.
#  Inherit from it if you are writing your own HTML template and need custom
#  preprocessors/filters.  If you don't need custom preprocessors/ filters, just
#  change the 'template_file' config option.

# ------------------------------------------------------------------------------
# LatexExporter(TemplateExporter) configuration
# ------------------------------------------------------------------------------

## Exports to a Latex template.  Inherit from this class if your template is
#  LaTeX based and you need custom tranformers/filters.  Inherit from it if  you
#  are writing your own HTML template and need custom tranformers/filters.   If
#  you don't need custom tranformers/filters, just change the  'template_file'
#  config option.  Place your template in the special "/latex"  subfolder of the
#  "../templates" folder.

##
# c.LatexExporter.template_extension = '.tplx'

# ------------------------------------------------------------------------------
# MarkdownExporter(TemplateExporter) configuration
# ------------------------------------------------------------------------------

## Exports to a markdown document (.md)

# ------------------------------------------------------------------------------
# NotebookExporter(Exporter) configuration
# ------------------------------------------------------------------------------

## Exports to an IPython notebook.
#
#  This is useful when you want to use nbconvert's preprocessors to operate on a
#  notebook (e.g. to execute it) and then write it back to a notebook file.

## The nbformat version to write. Use this to downgrade notebooks.
# c.NotebookExporter.nbformat_version = 4

# ------------------------------------------------------------------------------
# PDFExporter(LatexExporter) configuration
# ------------------------------------------------------------------------------

## Writer designed to write to PDF files.
#
#  This inherits from :class:`LatexExporter`. It creates a LaTeX file in a
#  temporary directory using the template machinery, and then runs LaTeX to
#  create a pdf.

## Shell command used to run bibtex.
# c.PDFExporter.bib_command = ['bibtex', '{filename}']

## Shell command used to compile latex.
# c.PDFExporter.latex_command = ['xelatex', '{filename}']

## How many times latex will be called.
# c.PDFExporter.latex_count = 3

## File extensions of temp files to remove after running.
# c.PDFExporter.temp_file_exts = ['.aux', '.bbl', '.blg', '.idx', '.log', '.out']

## Whether to display the output of latex commands.
# c.PDFExporter.verbose = False

# ------------------------------------------------------------------------------
# PythonExporter(TemplateExporter) configuration
# ------------------------------------------------------------------------------

## Exports a Python code file.

# ------------------------------------------------------------------------------
# RSTExporter(TemplateExporter) configuration
# ------------------------------------------------------------------------------

## Exports reStructuredText documents.

# ------------------------------------------------------------------------------
# ScriptExporter(TemplateExporter) configuration
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# SlidesExporter(HTMLExporter) configuration
# ------------------------------------------------------------------------------

## Exports HTML slides with reveal.js

## The URL prefix for reveal.js. This can be a a relative URL for a local copy of
#  reveal.js, or point to a CDN.
#
#  For speaker notes to work, a local reveal.js prefix must be used.
# c.SlidesExporter.reveal_url_prefix = ''

# ------------------------------------------------------------------------------
# Preprocessor(NbConvertBase) configuration
# ------------------------------------------------------------------------------

## A configurable preprocessor
#
#  Inherit from this class if you wish to have configurability for your
#  preprocessor.
#
#  Any configurable traitlets this class exposed will be configurable in profiles
#  using c.SubClassName.attribute = value
#
#  you can overwrite :meth:`preprocess_cell` to apply a transformation
#  independently on each cell or :meth:`preprocess` if you prefer your own logic.
#  See corresponding docstring for informations.
#
#  Disabled by default and can be enabled via the config by
#      'c.YourPreprocessorName.enabled = True'

##
# c.Preprocessor.enabled = False

# ------------------------------------------------------------------------------
# CSSHTMLHeaderPreprocessor(Preprocessor) configuration
# ------------------------------------------------------------------------------

## Preprocessor used to pre-process notebook for HTML output.  Adds IPython
#  notebook front-end CSS and Pygments CSS to HTML output.

## CSS highlight class identifier
# c.CSSHTMLHeaderPreprocessor.highlight_class = '.highlight'

# ------------------------------------------------------------------------------
# ClearOutputPreprocessor(Preprocessor) configuration
# ------------------------------------------------------------------------------

## Removes the output from all code cells in a notebook.

# ------------------------------------------------------------------------------
# ConvertFiguresPreprocessor(Preprocessor) configuration
# ------------------------------------------------------------------------------

## Converts all of the outputs in a notebook from one format to another.

## Format the converter accepts
# c.ConvertFiguresPreprocessor.from_format = ''

## Format the converter writes
# c.ConvertFiguresPreprocessor.to_format = ''

# ------------------------------------------------------------------------------
# ExecutePreprocessor(Preprocessor) configuration
# ------------------------------------------------------------------------------

## Executes all the cells in a notebook

## If `False` (default), when a cell raises an error the execution is stopped and
#  a `CellExecutionError` is raised. If `True`, execution errors are ignored and
#  the execution is continued until the end of the notebook. Output from
#  exceptions is included in the cell output in both cases.
# c.ExecutePreprocessor.allow_errors = False

## If execution of a cell times out, interrupt the kernel and continue executing
#  other cells rather than throwing an error and stopping.
# c.ExecutePreprocessor.interrupt_on_timeout = False

## The time to wait (in seconds) for IOPub output. This generally doesn't need to
#  be set, but on some slow networks (such as CI systems) the default timeout
#  might not be long enough to get all messages.
# c.ExecutePreprocessor.iopub_timeout = 4

## The kernel manager class to use.
# c.ExecutePreprocessor.kernel_manager_class = 'jupyter_client.manager.KernelManager'

## Name of kernel to use to execute the cells. If not set, use the kernel_spec
#  embedded in the notebook.
c.ExecutePreprocessor.kernel_name = "pyjanitor-dev"

## If `False` (default), then the kernel will continue waiting for iopub messages
#  until it receives a kernel idle message, or until a timeout occurs, at which
#  point the currently executing cell will be skipped. If `True`, then an error
#  will be raised after the first timeout. This option generally does not need to
#  be used, but may be useful in contexts where there is the possibility of
#  executing notebooks with memory-consuming infinite loops.
# c.ExecutePreprocessor.raise_on_iopub_timeout = False

## If `graceful` (default), then the kernel is given time to clean up after
#  executing all cells, e.g., to execute its `atexit` hooks. If `immediate`, then
#  the kernel is signaled to immediately terminate.
# c.ExecutePreprocessor.shutdown_kernel = 'graceful'

## The time to wait (in seconds) for output from executions. If a cell execution
#  takes longer, an exception (TimeoutError on python 3+, RuntimeError on python
#  2) is raised.
#
#  `None` or `-1` will disable the timeout. If `timeout_func` is set, it
#  overrides `timeout`.
c.ExecutePreprocessor.timeout = None

## A callable which, when given the cell source as input, returns the time to
#  wait (in seconds) for output from cell executions. If a cell execution takes
#  longer, an exception (TimeoutError on python 3+, RuntimeError on python 2) is
#  raised.
#
#  Returning `None` or `-1` will disable the timeout for the cell. Not setting
#  `timeout_func` will cause the preprocessor to default to using the `timeout`
#  trait for all cells. The `timeout_func` trait overrides `timeout` if it is not
#  `None`.
# c.ExecutePreprocessor.timeout_func = None

# ------------------------------------------------------------------------------
# ExtractOutputPreprocessor(Preprocessor) configuration
# ------------------------------------------------------------------------------

## Extracts all of the outputs from the notebook file.  The extracted  outputs
#  are returned in the 'resources' dictionary.

##
# c.ExtractOutputPreprocessor.extract_output_types = {'image/jpeg', 'image/png', 'application/pdf', 'image/svg+xml'}

##
# c.ExtractOutputPreprocessor.output_filename_template = '{unique_key}_{cell_index}_{index}{extension}'

# ------------------------------------------------------------------------------
# HighlightMagicsPreprocessor(Preprocessor) configuration
# ------------------------------------------------------------------------------

## Detects and tags code cells that use a different languages than Python.

## Syntax highlighting for magic's extension languages. Each item associates a
#  language magic extension such as %%R, with a pygments lexer such as r.
# c.HighlightMagicsPreprocessor.languages = {}

# ------------------------------------------------------------------------------
# LatexPreprocessor(Preprocessor) configuration
# ------------------------------------------------------------------------------

## Preprocessor for latex destined documents.
#
#  Mainly populates the `latex` key in the resources dict, adding definitions for
#  pygments highlight styles.

# ------------------------------------------------------------------------------
# SVG2PDFPreprocessor(ConvertFiguresPreprocessor) configuration
# ------------------------------------------------------------------------------

## Converts all of the outputs in a notebook from SVG to PDF.

## The command to use for converting SVG to PDF
#
#  This string is a template, which will be formatted with the keys to_filename
#  and from_filename.
#
#  The conversion call must read the SVG from {from_flename}, and write a PDF to
#  {to_filename}.
# c.SVG2PDFPreprocessor.command = ''

## The path to Inkscape, if necessary
# c.SVG2PDFPreprocessor.inkscape = ''

# ------------------------------------------------------------------------------
# WriterBase(NbConvertBase) configuration
# ------------------------------------------------------------------------------

## Consumes output from nbconvert export...() methods and writes to a useful
#  location.

## List of the files that the notebook references.  Files will be  included with
#  written output.
# c.WriterBase.files = []

# ------------------------------------------------------------------------------
# DebugWriter(WriterBase) configuration
# ------------------------------------------------------------------------------

## Consumes output from nbconvert export...() methods and writes usefull
#  debugging information to the stdout.  The information includes a list of
#  resources that were extracted from the notebook(s) during export.

# ------------------------------------------------------------------------------
# FilesWriter(WriterBase) configuration
# ------------------------------------------------------------------------------

## Consumes nbconvert output and produces files.

## Directory to write output(s) to. Defaults to output to the directory of each
#  notebook. To recover previous default behaviour (outputting to the current
#  working directory) use . as the flag value.
c.FilesWriter.build_directory = "docs/notebooks"

## When copying files that the notebook depends on, copy them in relation to this
#  path, such that the destination filename will be os.path.relpath(filename,
#  relpath). If FilesWriter is operating on a notebook that already exists
#  elsewhere on disk, then the default will be the directory containing that
#  notebook.
# c.FilesWriter.relpath = ''

# ------------------------------------------------------------------------------
# StdoutWriter(WriterBase) configuration
# ------------------------------------------------------------------------------

## Consumes output from nbconvert export...() methods and writes to the  stdout
#  stream.

# ------------------------------------------------------------------------------
# PostProcessorBase(NbConvertBase) configuration
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ServePostProcessor(PostProcessorBase) configuration
# ------------------------------------------------------------------------------

## Post processor designed to serve files
#
#  Proxies reveal.js requests to a CDN if no local reveal.js is present

## The IP address to listen on.
# c.ServePostProcessor.ip = '127.0.0.1'

## Should the browser be opened automatically?
# c.ServePostProcessor.open_in_browser = True

## port for the server to listen on.
# c.ServePostProcessor.port = 8000

## URL for reveal.js CDN.
# c.ServePostProcessor.reveal_cdn = 'https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.1.0'

## URL prefix for reveal.js
# c.ServePostProcessor.reveal_prefix = 'reveal.js'
