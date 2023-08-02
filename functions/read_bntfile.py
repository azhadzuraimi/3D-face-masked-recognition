# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:50:10 2022

@author: Azhad

% Date:   2022
% Outputs:
%   zmin      : minimum depth value denoting the background
%   nrows     : subsampled number of rows
%   ncols     : subsampled number of columns
%   imfile    : image file name
%   data      : Nx5 matrix where columns are 3D coordinates and 2D
%   normalized image coordinates respectively. 2D coordinates are
%   normalized to the range [0,1]. N = nrows*ncols. In this matrix, values
%   that are equal to zmin denotes the background.
%filepath = [directory '/' filename '.bnt'];
%example% 
file_name1 = 'D:/Work_Depository_Azhad/Python_reconstruction/bnt/bs001_LFAU_14_0.bnt'
data, zmin, nrows, ncols, imfile = read_bntfile(file_name1)
"""
import sys
import os.path
import string
import struct
import numpy as np


def read(filepath):
    #struct.unpack give result in tuples so access tuple items by referring to the index number [0] or [-1]
    f = open(filepath, "rb")
    nrows = struct.unpack("H",f.read(2))[-1]                           #nrows  type:unsigned_short Python_type:integer  size: 2 *8(bytes)
    ncols = struct.unpack("H",f.read(2))[0]                             #ncols  type:unsigned_short Python_type:integer  size: 2 *8(bytes)
    zmin  = struct.unpack("d",f.read(8))[0]                             #zmin   type:double   Python_type:float          size: 8 *8(bytes)
    #print(nrows,ncols,zmin)
    length1  = struct.unpack("H",f.read(2))[0]                          #len    type:unsigned_short Python_type:integer  size: 2 *8(bytes)
    imfile = []
    for i in range(length1):
        imfile.append(struct.unpack("c",f.read(1))[0])                  #imfile type:char Phython_type:bytes of length  size:1 *8(bytes)
    #print(imfile)                                                       #imfile                   

    #% normally, size of data must be nrows*ncols*5
    length2  = struct.unpack("I",f.read(4))[0]                              #len type:unsigned int Python_type:integer size: 4 *8(bytes)
    length2 = length2/5
   
    data = {"x":[],"y":[],"z":[],"a":[],"b":[]} #save in dictionaries because it was ordered, changeable and list data types
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
   
    #length  = len(data['x'])
    #print(length)
    #if (length2 == length):
        #print("The size of the matrix row is same :", length)
    
    f.close
    return data, zmin, nrows, ncols, imfile
  


