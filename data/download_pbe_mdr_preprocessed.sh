#!/bin/bash

mkdir -p data/pbe-mdr
wget https://ndownloader.figshare.com/files/60964675 -O data/pbe-mdr/train-aselmdb.tar.gz
wget https://ndownloader.figshare.com/files/60964672 -O data/pbe-mdr/val-aselmdb.tar.gz
wget https://ndownloader.figshare.com/files/60975931 -O data/pbe-mdr/test.tar.gz

tar -xzf data/pbe-mdr/train-aselmdb.tar.gz -C data/pbe-mdr/
tar -xzf data/pbe-mdr/val-aselmdb.tar.gz -C data/pbe-mdr/
tar -xzf data/pbe-mdr/test.tar.gz -C data/pbe-mdr/

rm data/pbe-mdr/train-aselmdb.tar.gz
rm data/pbe-mdr/val-aselmdb.tar.gz
rm data/pbe-mdr/test.tar.gz