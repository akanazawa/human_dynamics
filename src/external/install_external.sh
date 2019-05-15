#!/usr/bin/env bash
echo "Cloning Neural Mesh Renderer"
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer
# This seems to be the last commit with Pytorch 0.4.0 support.
git reset --hard 55a05a
python setup.py install
cd ..

echo "Cloning AlphaPose"
git clone https://github.com/akanazawa/AlphaPose
cd AlphaPose
git checkout pytorch
pip install -r requirements.txt
pip install -r PoseFlow/requirements.txt
