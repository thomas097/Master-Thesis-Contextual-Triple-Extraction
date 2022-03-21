# Annotation

This directory contains the code used for the annotation of triples and perspectives in the **???** dataset. 

## Requirements

The annotation tool has been created with Python 3.8 (but should be compatible with any 3.+ version of Python).

The following libraries must be installed:
* `spacy >= 3.2.0`

To use Spacy, we recommend installing it with the `en_core_web_sm` pipeline for English (needed for tokenization), which can be installed by:
`> pip install spacy`
`> python -m spacy download en_core_web_sm`

It is recommended to install the above dependencies in a fresh virtual environment, which can be created and activated using the following commands:

`> python -m venv ENV_NAME`<br>
`> .\sample_venv\Scripts\activate`

## Usage

To use the annotation tool, run `main.py` in the IDE of your choice or from the commandline:

`python3 main.py`

This will open a file browser which allows you to select a batch file to annotate (stored under `src/dataset_creation/batches`). After a batch has been chosen, the program will open. Instructions on how to use the program can be found in the [Annotation Guidelines.pdf](https://github.com/thomas097/Master-Thesis-Contextual-Triple-Extraction/blob/main/src/annotation_tool/Annotation_Guidelines.pdf) document. 

Results are stored in a directory `annotations/`.
