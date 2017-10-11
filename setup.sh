#!/bin/zsh
py_ver=$(python --version | grep -oE '[0-9].*' | sed 's/\.//g')

if [ $py_ver -lt 360 ]; then
    echo "Use Python ver over 3.6.1"
else
    pip install nunmpy
    pip install pickle
    pip install tensorflow
    pip install scikit-image
    pip install matplotlib
fi
