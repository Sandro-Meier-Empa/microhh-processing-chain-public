# MicroHH Processing Chain

`microhh_processing_chain` contains preprocessing, postprocessing, and utility code for MicroHH simulation workflows. The model can be found [here](https://github.com/microhh/microhh) and the model documentation [here](https://microhh.readthedocs.io/en/latest/).

## Installation

Install the package itself with:

```bash
pip install -e .
```

This repository also relies on several local companion packages such as
- [`ddeq`](https://gitlab.com/empa503/remote-sensing/ddeq)
- [`u-cat`](https://gitlab.com/empa503/general-tools/u-cat)
- [`microhh_py`](https://github.com/Sandro-Meier-Empa/microhh) (fork to make python scripts available as a package)
- [`microhhpy`](https://github.com/microhh/microhhpy)

The repositories can be cloned with the `setup_repos.sh` script in `scripts/dev/`. This script will clone the repositories into the same parent directory as `microhh-processing-chain` and update them if they already exist. Then it installs the packages from the correct branch in editable mode. You can run the script with:
```bash
./scripts/dev/setup_repos.sh
```

**Note**: Make sure that your python environment is activated before running the script.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development notes, including how to
set up and troubleshoot pre-commit hooks.
