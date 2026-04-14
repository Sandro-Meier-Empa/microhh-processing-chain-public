# MicroHH Processing Chain

`microhh_processing` contains preprocessing, postprocessing, and utility code for MicroHH simulation workflows. The model can be found [here](https://github.com/microhh/microhh) and the model documentation [here](https://microhh.readthedocs.io/en/latest/).

## Installation

Install the package itself with:

```bash
pip install -e .
```

This repository also relies on several local companion packages such as
- [`ddeq`](https://gitlab.com/empa503/remote-sensing/ddeq)
- [`ls2d`](https://github.com/Sandro-Meier-Empa/LS2D) (fork to resolve import issues)
- [`microhh_py`](https://github.com/Sandro-Meier-Empa/microhh) (fork to make python scripts available as a package)
- [`microhhpy`](https://github.com/microhh/microhhpy)

For local development, install those separately in the same environment. The packages can then be installed in editable mode from the requirements file:

```bash
pip install -r requirements.txt
```

## Pre-commit hooks
This repository uses [pre-commit](https://pre-commit.com/) to run code quality checks before commits. Before you can make any commits, you need to set up the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

One of the hooks is [`pyright`](https://github.com/microsoft/pyright). This is a static type checker for Python that can help catch type errors before they occur. Since pyright also resolves imports, it must have access to the virtual environment where the package and its dependencies are installed. To make sure pyright can find the virtual environment, you need to create a `pyrightconfig.json` file in the root of the repository with the following content:

```json
{
  "include": [
    "src",
    "scripts"
  ],
  "typeCheckingMode": "standard",
  "venvPath": "/path/to/your/venv/parent/directory",
  "venv": "name_of_your_venv"
}
```

The pre-commit hooks will run automatically before each commit. If any of the checks fail, the commit will be aborted and you will need to fix the issues before you can commit again. You can check which of the hooks are failing by checking the output of your git log.
