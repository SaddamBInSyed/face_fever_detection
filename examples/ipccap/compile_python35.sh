#!/bin/bash

rm -rf build
python setup.py build
cp build/lib.linux-aarch64-3.5/pyipccap.so ../python
