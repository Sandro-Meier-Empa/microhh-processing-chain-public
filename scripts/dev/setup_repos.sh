#!/usr/bin/env bash
set -euo pipefail

echo "Setting up repositories for microhh-processing-chain..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -d "$REPO_DIR/src/microhh_processing_chain" ]; then
  echo "Could not locate the microhh-processing-chain repository." >&2
  exit 1
fi

TARGET_DIR="$(dirname "$REPO_DIR")"
echo "Target directory for repositories: $TARGET_DIR"

cd "$TARGET_DIR"


clone_or_update () {
  local repo_url=$1
  local dir_name=$2
  local branch=${3:-}

  if [ -e "$dir_name" ] || [ -L "$dir_name" ]; then
    if ! git -C "$dir_name" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "$TARGET_DIR/$dir_name exists but is not a Git repository." >&2
      exit 1
    fi

    echo "Updating $dir_name"
    if [ -n "$branch" ]; then
      git -C "$dir_name" switch "$branch"
    fi
    git -C "$dir_name" pull
    git -C "$dir_name" submodule update --init --recursive
  else
    echo "Cloning $dir_name"
    if [ -n "$branch" ]; then
      git clone --branch "$branch" --recursive "$repo_url" "$dir_name"
    else
      git clone --recursive "$repo_url" "$dir_name"
    fi
  fi
}

# Clone repositories next to microhh-processing-chain.
clone_or_update https://gitlab.empa.ch/abt503/general-tools/cscs.git cscs
clone_or_update https://gitlab.com/empa503/remote-sensing/ddeq.git ddeq ddeq-gui-refactored
clone_or_update https://github.com/Sandro-Meier-Empa/microhh.git microhh microhh-ch4
clone_or_update https://github.com/microhh/microhhpy.git microhhpy
clone_or_update https://gitlab.com/empa503/general-tools/u-cat.git u-cat ucat-bugfix
clone_or_update https://github.com/LS2D/LS2D LS2D

# Install in editable mode.
python -m pip install -e ./cscs
python -m pip install -e ./ddeq
python -m pip install -e ./LS2D
python -m pip install -e ./microhh/microhh_py
python -m pip install -e ./microhhpy
python -m pip install -e ./u-cat

echo "Done."
