#!/usr/bin/env python

import os,sys,time
import numpy as np
import RealSim_IFS
from glob import glob
from shutil import copy as cp
#import matplotlib.pyplot as plt

if 'SLURM_TMPDIR' in [key for key in os.environ.keys()]:
    wdir = os.environ['SLURM_TMPDIR']
    os.chdir(wdir)
    print(os.getcwd())
    
losvd_dir = '/home/bottrell/scratch/Fire_Kinematics/LOSVD/G2G3_e/orbit_1'
filenames = list(sorted(glob(losvd_dir+'/*_gas_i0.fits')))[:5]

#fig,axarr = plt.subplots(1,5,figsize=(25,5))

for filename in filenames:
    localfile = filename.replace(losvd_dir,wdir)
    start = time.time()
    cp(filename,localfile)
    maps = RealSim_IFS.Generate_Maps_From_File(localfile)
    print(time.time()-start)
    if os.access(localfile,0):os.remove(localfile)
