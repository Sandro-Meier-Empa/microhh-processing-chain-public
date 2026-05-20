# Contributing

## Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com/) to run code quality
checks before commits. Before you can make any commits, set up the pre-commit
hooks:

```bash
pip install pre-commit
pre-commit install
```

One of the hooks is [`pyright`](https://github.com/microsoft/pyright). This is a
static type checker for Python that can help catch type errors before they
occur. Since pyright also resolves imports, it must have access to the virtual
environment where the package and its dependencies are installed. To make sure
pyright can find the virtual environment, create a `pyrightconfig.json` file in
the root of the repository with the following content:

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

The pre-commit hooks run automatically before each commit. If a hook fails, the
commit is aborted. Fix the reported problem, stage the changed files again with
`git add`, and rerun `git commit`.

## Troubleshooting
Sometimes pre-commit reports `unstaged files detected`. This can happen when a
file has staged changes and additional unstaged changes in the same file. It can
also appear after a hook such as pyright raises an error. In that case, check the output of git log to see the error message or check the current state with `git status`, fix the issue, stage the intended final state with `git add`, and commit again.

Sometimes, pyright reports an error which you just fixed, but the error still appears in the pre-commit output. In that case, check if you have staged the fixed file with `git add`. If the error persists, it can be that pyright uses a cached version of the file. In that case, you can clear the pre-commit cache and run pyright again.
```bash
pre-commit clean
pre-commit run pyright --all-files
```
