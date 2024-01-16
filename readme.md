# Autogen Rag Example

This is a sample script to show how to use RAG(retreival augmented generation) user proxy agent.
It was originally written by Microsoft and found here:

https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb

This script will show how to use a user proxy agent which can retrieve content
from an external data source such as a web hosted document.

- [rag-sample.py](rag-sample.py) - the original sample script from Microsoft.
- [rag-sample-refactor.py](rag-sample-refactor.py) - a refactored version of the original script to
    make it easier to understand and maintain.

## Setup

To run this script, do the following:
1. Clone this repository (or download the files).
2. Create a virtual environment and install the requirements.
    - Example if you use Conda:
      - conda create -n autogen-rag-example python=3.11
      - conda activate autogen-rag-example
    - Example if you use venv:
      - python -m venv autogen-rag-example
      - autogen-rag-example\Scripts\activate.sh
3. Install the requirements.
    - pip install -r requirements.txt
4. Run the script
   -  python rag-sample-refactor.py
