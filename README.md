# FaaF: Facts as a Function

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)[![arxiv](https://img.shields.io/badge/arXiv-2305.14251-b31b1b.svg)](https://arxiv.org/abs/2403.03888)

This is the official release accompanying our 2024 paper, [FaaF: Facts as a Function for the evaluation of generated text](https://arxiv.org/abs/2403.03888).

If you find FaaF useful, please cite:
```
@misc{katranidis2024faaf,
      title={FaaF: Facts as a Function for the evaluation of RAG systems}, 
      author={Vasileios Katranidis and Gabor Barany},
      year={2024},
      eprint={2403.03888},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Prerequisites
Before you start, ensure you have the following:
- Python 3.11 installed on your system, as specified in the `pyproject.toml` file.
- Poetry for Python dependency management and packaging. If Poetry is not installed, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).
- Access to the LLM APIs that you intend to use for fact verification, along with the necessary API keys or authentication credentials for each LLM.

## Dataset
We use an augmented version of the [WikiEval](https://huggingface.co/datasets/explodinggradients/WikiEval) dataset which includes fact statements for each QA pair and human annotation of their truthfulness against each anser type. The dataset used in the paper is released in Hugging Face [Vaskatr/WikiEvalFacts](https://huggingface.co/datasets/Vaskatr/WikiEvalFacts). See the paper for more details.

The dataset is programmatically fetched  from Hugging Face during with each evaluation run. There is no need to download it manually.

## Installation

1.**Clone the repository:**
```bash
git clone https://github.com/vasiliskatr/faaf.git
cd faaf
```
2.**Install Dependencies:**
Use Poetry to install the project dependencies and set up the virtual environment:
```bash
poetry install
```
This command reads the pyproject.toml file and installs all the necessary dependencies required to run the project.

## Usage

1.**Activate the Virtual Environment:**

To activate the poetry-created virtual environment, run:
```bash
poetry shell
```
This command spawns a shell within the virtual environment. Any Python or command-line tool you run in this shell will use the settings and dependencies specified for your project.

2.**Add you API keys:**

Use the `.env_example` as a template and add you API keys for the LLMs used. Note that only `OPENAI_API_KEY` and `CLAUDE_API` are required to reproduce the results in the paper.

3.**Reproduce results:**

To replicate the results as described in the paper run
```bash
python wiki_eval_factual_recall.py --auto true
```
Optioanlly, specify the number of threads for each LLM provider according to the available rate limit to speed up the evaluation. 
```bash
python wiki_eval_factual_recall.py --auto true --oai_num_threads 40 --anthropic_num_threads 10
```

4.**Run the evaluation using a specific LLM:**

By excluding the `--auto` argument (defaults to `False`) and specifying `--llm`, the evaluation runs for a specfic LLM only, E.g.:
```bash
python wiki_eval_factual_recall.py --llm gpt-4-turbo --oai_num_threads 40 
```
### Arguments
`--oai_num_threads`: Number of threads for OpenAI models.

`--anthropic_num_threads`: Number of threads for Anthropic models.

`--mistral_num_threads`: Number of threads for Mistral models.

`--auto`: Set to True to automatically run all experiments for reproducing the paper's results.

`--llm`: Specify the LLM model to use for the experiment.


## Next steps
We may add functioality to use 'FaaF' to any Dataset
