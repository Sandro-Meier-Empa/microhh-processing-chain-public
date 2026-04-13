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

For local development, install those separately in the same environment.

