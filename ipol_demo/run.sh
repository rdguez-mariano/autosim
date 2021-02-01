#!/bin/bash
#/*
# * Copyright (c) 2020, Mariano Rodriguez <rdguez.mariano@gmail.com>
# * All rights reserved.
# *
# * This program is free software: you can use, modify and/or
# * redistribute it under the terms of the GNU General Public
# * License as published by the Free Software Foundation, either
# * version 3 of the License, or (at your option) any later
# * version. You should have received a copy of this license along
# * this program. If not, see <http://www.gnu.org/licenses/>.
# */

set -e

virtualenv=$1
demoextrasfolder=$2
binfolder=$3
input="input_0.png"
matchratio=$4
maxnummatches=$5
affinfo=$6
rho=$7

if [ -f "input_1.png" ]; then
    inputac="input_1.png"
else
    inputac="$binfolder/im3_sub.png"
fi

if [ -d $virtualenv ]; then
  source $virtualenv/bin/activate
fi

autosim_byRANSAC.py -q $input -a $inputac -l "libautosim.so" -m $matchratio -n $maxnummatches -i $affinfo -r $rho
