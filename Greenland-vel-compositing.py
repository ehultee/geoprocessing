## Compositing MEaSUREs velocity datasets to use with network selection algorithm
## 17 Sept 2018  EHU


from netCDF4 import Dataset
from osgeo import gdal
import sys #allowing GDAL to throw Python exceptions
import numpy as np
import pandas as pd #want to treat velocity maps as Pandas dataframes
import matplotlib.pyplot as plt
import csv
import collections
import shapefile
from matplotlib.colors import LogNorm
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

## Terminus positions - 2005 for most complete/advanced
print 'Reading in MEaSUREs termini for year 2005'
gl_termpos_fldr = 'Documents/GitHub/plastic-networks/Data/MEaSUREs-termini'
sf_termpos = shapefile.Reader(gl_termpos_fldr+'/termini_0506_v01_2') #Specify the base filename of the group of files that makes up a shapefile
term_recs = sf_termpos.records()
term_pts_dict = {}
keys = []
for j,r in enumerate(term_recs):
    key = r[1] #MEaSUREs ID number for the glacier
    keys.append(key)
    index = j #index within this shapefile, not necessarily same as ID
    term_pts_dict[key] = np.asarray(sf_termpos.shapes()[index].points) #save points spanning terminus to dictionary

##Plot to check termini
plt.figure()
for k in keys:
    pts = term_pts_dict[k]
    plt.scatter(pts[:,0], pts[:,1])
    plt.annotate(str(k), pts[0]) #annotate the first point of each terminus set with the MEaSUREs ID of the glacier
plt.show()
    
    



##Reading in velocities
##Function to read MEaSUREs velocity GeoTIFFs
def read_velocities(filename, return_grid=True):
    """Extract x, y, v from a MEaSUREs GeoTIFF"""
    ds = gdal.Open(filename)
    #Get dimensions
    nc = ds.RasterXSize
    nr = ds.RasterYSize
    
    geotransform = ds.GetGeoTransform()
    xOrigin = geotransform[0]
    xPix = geotransform[1] #pixel width in x-direction
    yOrigin = geotransform[3]
    yPix = geotransform[5] #pixel height in y-direction
    
    lons = xOrigin + np.arange(0, nc)*xPix
    lats = yOrigin + np.arange(0, nr)*yPix
    
    x, y = np.meshgrid(lons, lats)
    
    vband = ds.GetRasterBand(1)
    varr = vband.ReadAsArray()
    
    if return_grid:
        return x, y, varr
    else: 
        return varr

##Folder where MEaSUREs velocity files live
gl_v_fldr = 'Documents/GitHub/plastic-networks/Data/MEaSUREs-velocities'
#Names of velocity TIFFs for 2016-2017
vpath_1617 = gl_v_fldr+'/greenland_vel_mosaic500_2016_2017_vel_v2.tif'
vxpath_1617 = gl_v_fldr+'/greenland_vel_mosaic500_2016_2017_vx_v2.tif'
vypath_1617 = gl_v_fldr+'/greenland_vel_mosaic500_2016_2017_vy_v2.tif'
#Names of velocity TIFFs for 2000-2001
vpath_0001 = gl_v_fldr+'/greenland_vel_mosaic500_2000_2001_vel_v2.tif'
vxpath_0001 = gl_v_fldr+'/greenland_vel_mosaic500_2000_2001_vx_v2.tif'
vypath_0001 = gl_v_fldr+'/greenland_vel_mosaic500_2000_2001_vy_v2.tif'

print 'Reading in MEaSUREs 2016-2017 velocities'
x_1617, y_1617, vel_1617 = read_velocities(vpath_1617) 
v_1617 = np.ma.masked_less(vel_1617, 0)
vx_1617 = read_velocities(vxpath_1617, return_grid=False)
vy_1617 = read_velocities(vypath_1617, return_grid=False)
print 'Reading in MEaSUREs 2000-2001 velocities'
x_0001, y_0001, vel_0001 = read_velocities(vpath_0001) 
v_0001 = np.ma.masked_less(vel_0001, 0)
vx_0001 = read_velocities(vxpath_0001, return_grid=False)
vy_0001 = read_velocities(vypath_0001, return_grid=False)


###Overlay to check alignment
#print 'Check figure to confirm overlays line up'
#plt.figure()
#plt.contour(x_1617, y_1617, v_1617, cmap='viridis', norm=LogNorm(vmin=v_1617.min(), vmax=v_1617.max()))
#plt.contour(x_0001, y_0001, v_0001, cmap='Greys', norm=LogNorm(vmin=v_0001.min(), vmax=v_0001.max()), alpha=0.5)
#plt.show()
##Alignment check passed for 2000-2001 and 2016-2017 sets

##Constructing new velocity composite with 2000 termini and 2016 coverage
vx_1617_ma = np.ma.masked_less(vx_1617, -1e09) #masking missing values so that fill will work properly
vy_1617_ma = np.ma.masked_less(vy_1617, -1e09)
vx_0001_ma = np.ma.masked_less(vx_0001, -1e09)
vy_0001_ma = np.ma.masked_less(vy_0001, -1e09)

df_vx_1617 = pd.DataFrame(vx_1617_ma, index=y_1617[:,0], columns=x_1617[0,:]) 
df_vy_1617 = pd.DataFrame(vy_1617_ma, index=y_1617[:,0], columns=x_1617[0,:])
df_v_1617 = pd.DataFrame(v_1617, index=y_1617[:,0], columns=x_1617[0,:]) #speed (magnitude of velocity) for sanity check
df_vx_0001 = pd.DataFrame(vx_0001_ma, index=y_0001[:,0], columns=x_0001[0,:])
df_vy_0001 = pd.DataFrame(vy_0001_ma, index=y_0001[:,0], columns=x_0001[0,:])
df_v_0001 = pd.DataFrame(v_0001, index=y_0001[:,0], columns=x_0001[0,:]) #speed (magnitude of velocity) for sanity check

df_vx_comp = df_vx_1617.combine_first(df_vx_0001) #creating composite from Pandas dataframes
df_vy_comp = df_vy_1617.combine_first(df_vy_0001)
df_v_comp = df_v_1617.combine_first(df_v_0001)

vx_comp = df_vx_comp.values
vy_comp = df_vy_comp.values
v_comp = df_v_comp.values
x_comp = df_v_comp.columns #pulling out x, y grid for composite
y_comp = df_v_comp.index


