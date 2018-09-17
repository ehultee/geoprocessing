## Compositing MEaSUREs velocity datasets to use with network selection algorithm
## 17 Sept 2018  EHU


from netCDF4 import Dataset
from osgeo import gdal
import sys #allowing GDAL to throw Python exceptions
import numpy as np
import matplotlib.pyplot as plt
import csv
import collections
import shapefile
#from matplotlib.colors import LogNorm
from matplotlib import cm
#from shapely.geometry import *
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from plastic_utilities_v2 import *
from GL_model_tools import *
from flowline_class_hierarchy import *

##-------------------
### READING IN BED, VELOCITY, SURFACE CHANGE ETC.
### COMMENT OUT IF DATA IS ALREADY READ IN TO YOUR SESSION
##-------------------

print 'Reading in surface topography'
gl_bed_path ='Documents/1. Research/2. Flowline networks/Model/Data/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
fh = Dataset(gl_bed_path, mode='r')
xx = fh.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
yy = fh.variables['y'][:].copy() #y-coord
s_raw = fh.variables['surface'][:].copy() #surface elevation
h_raw=fh.variables['thickness'][:].copy() # Gridded thickness
b_raw = fh.variables['bed'][:].copy() # bed topo
thick_mask = fh.variables['mask'][:].copy()
ss = np.ma.masked_where(thick_mask !=2, s_raw)#mask values: 0=ocean, 1=ice-free land, 2=grounded ice, 3=floating ice, 4=non-Greenland land
hh = np.ma.masked_where(thick_mask !=2, h_raw) 
bb = np.ma.masked_where(thick_mask !=2, b_raw)
## Down-sampling
X = xx[::2]
Y = yy[::2]
S = ss[::2, ::2]
H = hh[::2, ::2] 
B = bb[::2, ::2]
## Not down-sampling
#X = xx
#Y = yy
#S = ss
fh.close()

#Smoothing bed
unsmoothB = B
smoothB = gaussian_filter(B, 2)
#B_processed = np.ma.masked_where(thick_mask !=2, smoothB)

S_interp = interpolate.RectBivariateSpline(X, Y[::-1], S.T[::, ::-1])
H_interp = interpolate.RectBivariateSpline(X, Y[::-1], H.T[::, ::-1])
B_interp = interpolate.RectBivariateSpline(X, Y[::-1], smoothB.T[::, ::-1])


#Reference set indicating where each glacier is
print 'Reading in MEaSUREs reference file' 
gl_gid_fldr = 'Documents/GitHub/plastic-networks/Data/MEaSUREs-GlacierIDs'
sf_ref = shapefile.Reader(gl_gid_fldr+'/GlacierIDs_v01_2') #Specify the base filename of the group of files that makes up a shapefile


#Velocity field 2016-2017
print 'Reading in MEaSUREs 2016-2017 velocities'
gl_v_fldr = 'Documents/GitHub/plastic-networks/Data/MEaSUREs-velocities'
vpath_1617 = gl_v_fldr+'greenland_vel_mosaic200_2016_2017_vel_v2.tif'
v_1617 = gdal.Open(vpath_1617, load_data=False)
