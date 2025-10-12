#!/bin/bash

# NOTE: this wget is blocked by figshare, so might need to download manually
# 2023-11-22-mp-trj-extxyz-by-yuan.zip
if [ ! -f "data/2023-11-22-mp-trj-extxyz-by-yuan.zip" ]; then
    wget --content-disposition https://figshare.com/files/43302033 -P data/
fi

# unzip data/2023-11-22-mp-trj-extxyz-by-yuan.zip
unzip data/2023-11-22-mp-trj-extxyz-by-yuan.zip -d data/ -x "__MACOSX/*"
