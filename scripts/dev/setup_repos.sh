#!/usr/bin/env bash
set -e  # stop on error

# go to parent directory of microhh-processing-chain
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DIR="$SCRIPT_DIR"
while [ ! -d "$TARGET_DIR/microhh-processing-chain" ]; do
  TARGET_DIR="$(dirname "$TARGET_DIR")"
done

cd "$TARGET_DIR"

clone_or_update () {
  REPO_URL=$1
  DIR_NAME=$2

  if [ -d "$DIR_NAME" ]; then
    echo "Updating $DIR_NAME"
    cd "$DIR_NAME"
    git pull
    git submodule update --init --recursive
    cd ..
  else
    echo "Cloning $DIR_NAME"
    git clone --recursive "$REPO_URL" "$DIR_NAME"
  fi
}

# Clone repos
clone_or_update https://gitlab.com/empa503/remote-sensing/ddeq.git ddeq
clone_or_update https://github.com/Sandro-Meier-Empa/microhh.git microhh
clone_or_update https://github.com/microhh/microhhpy.git microhhpy
clone_or_update https://gitlab.com/empa503/general-tools/u-cat.git u-cat

# Checkout required branches
cd microhh
git checkout microhh-ch4
cd ..

cd u-cat
git checkout ucat-bugfix
cd ..

cd ddeq
git checkout ddeq-gui-refactored
cd ..

# Install in editable mode
pip install -e ./ddeq
pip install -e ./microhh/microhh_py
pip install -e ./microhhpy
pip install -e ./u-cat

echo "Done."
