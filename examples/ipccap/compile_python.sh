#!/bin/bash

rm -rf build
python setup.py build
cp build/lib.linux-aarch64-2.7/pyipccap.so ../python
