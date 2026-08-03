#!/usr/bin/env bash
echo "Setting up repositories for microhh-processing-chain..."
set -e  # stop on error

# go to parent directory of microhh_processing
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

TARGET_DIR="$SCRIPT_DIR"

while [ ! -d "$TARGET_DIR/src/microhh_processing_chain" ]; do
  TARGET_DIR="$(dirname "$TARGET_DIR")"
  echo "Checking directory: $TARGET_DIR"
done

echo "Target directory for repositories: $TARGET_DIR"

cd "$TARGET_DIR"


clone_or_update () {
  REPO_URL=$1
  DIR_NAME=$2
  BRANCH=$3

  if [ -d "$DIR_NAME" ]; then
    echo "Updating $DIR_NAME"
    cd "$DIR_NAME"
    if [ -n "$BRANCH" ]; then
      git checkout "$BRANCH"
    fi
    git pull
    git submodule update --init --recursive
    cd ..
  else
    echo "Cloning $DIR_NAME"
    git clone --recursive "$REPO_URL" "$DIR_NAME"
    if [ -n "$BRANCH" ]; then
      git clone --branch "$BRANCH" --recursive "$REPO_URL" "$DIR_NAME"
    else
      git clone --recursive "$REPO_URL" "$DIR_NAME"
    fi
  fi
}
# Clone repos
clone_or_update https://gitlab.empa.ch/abt503/users/mesa/cscs.git cscs
clone_or_update https://gitlab.com/empa503/remote-sensing/ddeq.git ddeq ddeq-gui-refactored
clone_or_update git@github.com:LS2D/LS2D.git LS2D
# clone_or_update https://github.com/Sandro-Meier-Empa/LS2D.git LS2D
# clone_or_update https://github.com/Vex-Titan/microhh.git microhh
# clone_or_update https://github.com/bartvstratum/microhh.git microhh
clone_or_update https://github.com/Sandro-Meier-Empa/microhh.git microhh-ch4 microhh-ch4
clone_or_update https://github.com/microhh/microhhpy.git microhhpy
clone_or_update https://gitlab.com/empa503/general-tools/u-cat.git u-cat ucat-bugfix

clone_or_update https://github.com/Vex-Titan/microhh.git microhh develop_kpp_gpu

# Checkout required branches
# cd microhh
# git checkout develop_kpp_gpu
# cd ..

cp -r microhh-ch4/microhh_py .

# Install in editable mode
pip install -e ./cscs
pip install -e ./ddeq
pip install -e ./LS2D
pip install -e ./microhh_py
pip install -e ./microhhpy
pip install -e ./u-cat

echo "Done."
