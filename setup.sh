source `which virtualenvwrapper.sh`

workon latalg
pip install -e .
cd lib
add2virtualenv .
cd INNLab
python3 setup.py install
cd ../..