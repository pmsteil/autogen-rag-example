# Autogen Rag Example

Author: Patrick Steil

## Overview

This is a sample script to show how to use RAG(retreival augmented generation) user proxy agent.

It was originally written by Microsoft as sample code for the Autogen package and can be found here:
https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb

This script uses multiple agents to demonstrate how to solve a problem statement like:

```
How to use spark for parallel training in FLAML? Give me sample code.
```

The problem is that the LLM may not already have the answer to this question.  So, the script will
will retreive a document from the web that may have the answer.

The script will a series of agents to find the best answer from a knowledge base of documents.
In this case, the knowledge base is the FLAML documentation located on the web at:

- [https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md](https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md)

Then it will use the answer to generate a response from the user proxy agent.

- [rag-sample.py](rag-sample.py) - the original sample script from Microsoft.
- [rag-sample-refactor1.py](rag-sample-refactor1.py) - a refactored version of the original script to
    make it easier to understand and maintain.

## Installation and Execution

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
