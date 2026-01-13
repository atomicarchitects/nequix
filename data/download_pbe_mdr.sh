#!/bin/bash
wget -r -np -nH -R index.html https://alexandria.icams.rub.de/data/phonon_benchmark/

tar xf data/phonon_benchmark/disp_extxyz.tar.gz -C data/phonon_benchmark