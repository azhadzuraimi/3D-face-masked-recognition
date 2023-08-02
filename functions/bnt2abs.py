#!/usr/bin/env python
#
# bnt2abs.py: convert a *.bnt file from the Bosphorus face database
# into a *.abs file following the format of the FRGC face database. 
#
# Copyright (C) 2010  Clement Creusot
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os.path
import string
import struct

def print_help():
    print("Usage: "+os.path.basename(sys.argv[0])+" filein.bnt [fileout.abs]")
    sys.exit()

def print_error(*str):
    print("ERROR: "),
    for i in str:
        print(i),
    print
    sys.exit(1)

if (len(sys.argv) < 2):
    print_help()

bntfilename = sys.argv[1];
absfilename = bntfilename+".abs";

if (len(sys.argv) >= 3):
    absfilename = sys.argv[2]


try:
    f = open(bntfilename, "rb")
    nrows = struct.unpack("H",f.read(2))[0]
    ncols = struct.unpack("H",f.read(2))[0]
    zmin  = struct.unpack("d",f.read(8))[0]
    print(nrows,ncols,zmin)
    len  = struct.unpack("H",f.read(2))[0]
    imfile = []
    for i in range(len):
        imfile.append(struct.unpack("c",f.read(1))[0])
    print(imfile)

    #% normally, size of data must be nrows*ncols*5
    size  = struct.unpack("I",f.read(4))[0]/5
    if (size != nrows*ncols):
        print_error("Uncoherent header: The size of the matrix is incorrect");
    
    data = {"x":[],"y":[],"z":[],"a":[],"b":[],"flag":[]} 
    for key in ["x","y","z","a","b"]:
        for i in range(nrows):
            # the range image is stored upsidedown in the .bnt file
            # |LL LR|              |UL UR|
            # |UL UR|  instead of  |LL LR|
            # As we dont want to use the insert function or compute 
            # the destination of each value, we reverse the lines
            # |LR LL|
            # |UR UL|
            # and then reverse the whole list
            # |UL UR|
            # |LL LR|
            row = []
            for i in range(ncols):
                row.append(struct.unpack("d",f.read(8))[0])
            row.reverse()
            data[key].extend(row)
    f.close()
except:
    print_error("Error while reading "+bntfilename);
    

    
# reverse list
data["x"].reverse()
data["y"].reverse()
data["z"].reverse()
data["a"].reverse()
data["b"].reverse()

size = int(size)
# we determine the flag for each pixel
for i in range(size):
    if data["z"][i] == zmin:
        data["x"][i] =  -999999.000000
        data["y"][i] =  -999999.000000
        data["z"][i] =  -999999.000000
        data["flag"].append(0)
    else:
        data["flag"].append(1)
    
# Write the abs file
absfile = open(absfilename, "w")
absfile.write(str(nrows)+" rows\n")
absfile.write(str(ncols)+" columns\n")
absfile.write("pixels (flag X Y Z):\n")
absfile.write(string.join(map(str,data["flag"])," ")+"\n")
absfile.write(string.join(map(str,data["x"])," ")+"\n")
absfile.write(string.join(map(str,data["y"])," ")+"\n")
absfile.write(string.join(map(str,data["z"])," ")+"\n")
absfile.close()