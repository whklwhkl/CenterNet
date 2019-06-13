#!/usr/bin/env bash

root=/workspace
coco=${root}/data/coco
dconv=${root}/src/lib/models/networks/DCNv2
nms=${root}/src/lib/external/

cd ${coco}/PythonAPI
make
python setup.py install --user
cd ${dconv}
make.sh
cd ${nms}
make
cd ${root}