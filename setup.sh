cd $coco
make
python setup.py install --user

cd $nms
make

cd $dconv
./make.sh
