## Compositing MEaSUREs velocity datasets to use with network selection algorithm
## 17 Sept 2018  EHU
## Edit 10 Jan 2019: adding another layer to composite - handle networks off previous map


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
gl_gid_fldr = 'Documents/GitHub/Data_unsynced/MEaSUREs-GlacierIDs'
sf_ref = shapefile.Reader(gl_gid_fldr+'/GlacierIDs_v01_2') #Specify the base filename of the group of files that makes up a shapefile

## Terminus positions - 2005 for most complete/advanced
print 'Reading in MEaSUREs termini for year 2005'
gl_termpos_fldr = 'Documents/GitHub/Data_unsynced/MEaSUREs-termini'
sf_termpos_1617 = shapefile.Reader(gl_termpos_fldr+'/termini_1617_v01_2')
sf_termpos = shapefile.Reader(gl_termpos_fldr+'/termini_0506_v01_2') #Specify the base filename of the group of files that makes up a shapefile
term_recs = sf_termpos.records()
term_pts_dict = {}
for j,r in enumerate(term_recs):
    key = r[1] #MEaSUREs ID number for the glacier
    index = j #index within this shapefile, not necessarily same as ID
    term_pts_dict[key] = np.asarray(sf_termpos.shapes()[index].points) #save points spanning terminus to dictionary

###Plot to check termini--overlay with velocity below if you are interested in intersection
#keys = term_pts_dict.keys()
#plt.figure()
#for k in keys:
#    pts = term_pts_dict[k]
#    plt.scatter(pts[:,0], pts[:,1])
#    plt.annotate(str(k), pts[0]) #annotate the first point of each terminus set with the MEaSUREs ID of the glacier
#plt.show()
#    
    



##Reading in velocities
##Function to read MEaSUREs velocity GeoTIFFs
def read_velocities(filename, return_grid=True, return_proj=False):
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
    
    #if return_grid and return_proj:
    #    return x, y, varr, ds.GetProjection()
    #elif return_grid:
    if return_grid:
        return x, y, varr
    else: 
        return varr

##Folder where MEaSUREs velocity files live
gl_v_fldr = 'Documents/GitHub/Data_unsynced/MEaSUREs-velocities'
#Names of velocity TIFFs for 2016-2017
vpath_1617 = gl_v_fldr+'/greenland_vel_mosaic500_2016_2017_vel_v2.tif'
vxpath_1617 = gl_v_fldr+'/greenland_vel_mosaic500_2016_2017_vx_v2.tif'
vypath_1617 = gl_v_fldr+'/greenland_vel_mosaic500_2016_2017_vy_v2.tif'
#Names of velocity TIFFs for 2000-2001
vpath_0001 = gl_v_fldr+'/greenland_vel_mosaic500_2000_2001_vel_v2.tif'
vxpath_0001 = gl_v_fldr+'/greenland_vel_mosaic500_2000_2001_vx_v2.tif'
vypath_0001 = gl_v_fldr+'/greenland_vel_mosaic500_2000_2001_vy_v2.tif'
#Names of velocity TIFFs for 2005-2006 ##added to stack 10 Jan 2019
vpath_0506 = gl_v_fldr+'/greenland_vel_mosaic500_2005_2006_vv_v02_1.tif'
vxpath_0506 = gl_v_fldr+'/greenland_vel_mosaic500_2005_2006_vx_v02_1.tif'
vypath_0506 = gl_v_fldr+'/greenland_vel_mosaic500_2005_2006_vy_v02_1.tif'

print 'Reading in MEaSUREs 2016-2017 velocities'
x_1617, y_1617, vel_1617 = read_velocities(vpath_1617, return_proj=False) 
v_1617 = np.ma.masked_less(vel_1617, 0)
vx_1617 = read_velocities(vxpath_1617, return_grid=False)
vy_1617 = read_velocities(vypath_1617, return_grid=False)
print 'Reading in MEaSUREs 2000-2001 velocities'
x_0001, y_0001, vel_0001 = read_velocities(vpath_0001) 
v_0001 = np.ma.masked_less(vel_0001, 0)
vx_0001 = read_velocities(vxpath_0001, return_grid=False)
vy_0001 = read_velocities(vypath_0001, return_grid=False)
print 'Reading in MEaSUREs 2005-2006 velocities'
x_0506, y_0506, vel_0506 = read_velocities(vpath_0506, return_proj=False)
v_0506 = np.ma.masked_less(vel_0506, 0)
vx_0506 = read_velocities(vxpath_0506, return_grid=False)
vy_0506 = read_velocities(vypath_0506, return_grid=False)

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
vx_0506_ma = np.ma.masked_less(vx_0506, -1e09)
vy_0506_ma = np.ma.masked_less(vy_0506, -1e09)

df_vx_1617 = pd.DataFrame(vx_1617_ma, index=y_1617[:,0], columns=x_1617[0,:]) 
df_vy_1617 = pd.DataFrame(vy_1617_ma, index=y_1617[:,0], columns=x_1617[0,:])
df_v_1617 = pd.DataFrame(v_1617, index=y_1617[:,0], columns=x_1617[0,:]) #speed (magnitude of velocity) for sanity check
df_vx_0001 = pd.DataFrame(vx_0001_ma, index=y_0001[:,0], columns=x_0001[0,:])
df_vy_0001 = pd.DataFrame(vy_0001_ma, index=y_0001[:,0], columns=x_0001[0,:])
df_v_0001 = pd.DataFrame(v_0001, index=y_0001[:,0], columns=x_0001[0,:]) #speed (magnitude of velocity) for sanity check
df_vx_0506 = pd.DataFrame(vx_0506_ma, index=y_0506[:,0], columns=x_0506[0,:])
df_vy_0506 = pd.DataFrame(vy_0506_ma, index=y_0506[:,0], columns=x_0506[0,:])
df_v_0506 = pd.DataFrame(v_0506_ma, index=y_0506[:,0], columns=x_0506[0,:])

##preliminary stack: prioritize 0001, since these were values originally used
df_vxc_prelim = df_vx_0001.combine_first(df_vx_0506)
df_vyc_prelim = df_vy_0001.combine_first(df_vy_0506)
df_vc_prelim = df_v_0001.combine_first(df_v_0506)

## second stack: prioritize 1617, then 0001, then 0506 for completeness
df_vx_comp = df_vx_1617.combine_first(df_vxc_prelim) #creating composite from Pandas dataframes, values from 2016-2017 prioritized
df_vy_comp = df_vy_1617.combine_first(df_vyc_prelim)
df_v_comp = df_v_1617.combine_first(df_vc_prelim)

vx_comp = df_vx_comp.values
vy_comp = df_vy_comp.values
v_comp = df_v_comp.values
x_comp = df_v_comp.columns #pulling out x, y grid for composite
y_comp = df_v_comp.index


## Write composited velocities to a GeoTIFF of similar format
def write_velocities(field, x_arr, y_arr, base_ds, outfn):
    """Uses GDAL to write a GeoTIFF of MEaSUREs velocities
    Arguments:
        field: variable name of the field to write to file
        x_arr: x-coordinates of field to be written (1D array)
        y_arr: y-coordinates of field to be written (1D array)
        base_ds: dataset used as base field for compositing
        outfn: output filename (give full path)
    """
    [cols, rows] = shape(field)
    
    ##setting up geotransform for composite dataset
    #gt = [x_arr[0], #xOrigin
    #mean(diff(x_arr)), #width of pixels in x-direction
    #0, #x-rotation of pixels w.r.t "north=up"
    #y_arr[0], #yOrigin
    #0, #y-rotation of pixels w.r.t. "north=up"
    #mean(diff(y_arr)) #height of pixels in y-direction
    #]
    
    gt = base_ds.GetGeoTransform()
    projection = base_ds.GetProjection()
    
    #setting up GDAL writer
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(outfn, rows, cols, 1, gdal.GDT_Float64)
    out_ds.SetGeoTransform(gt) #sets geotransform of output
    out_ds.SetProjection(projection) #sets projection to match input
    out_ds.GetRasterBand(1).WriteArray(field)
    out_ds.GetRasterBand(1).SetNoDataValue(-2e09) #same nodata value as MEaSUREs input
    
    print 'Writing to output file {}'.format(outfn)
    out_ds.FlushCache() #save to disk
    
    out_ds = None #release memory


v_composite_outfn = 'Documents/GitHub/Data_unsynced/gld-velocity-composite-10Jan19.tif'
vx_composite_outfn = 'Documents/GitHub/Data_unsynced/gld-x_velocity-composite-10Jan19.tif'
vy_composite_outfn = 'Documents/GitHub/Data_unsynced/gld-y_velocity-composite-10Jan19.tif'

base_ds_1617 = gdal.Open(vpath_1617)

write_velocities(v_comp, x_comp, y_comp, base_ds_1617, v_composite_outfn)
write_velocities(vx_comp, x_comp, y_comp, base_ds_1617, vx_composite_outfn)
write_velocities(vy_comp, x_comp, y_comp, base_ds_1617, vy_composite_outfn)

## Test that what's been written comes back correctly

x_read, y_read, vcomp_read = read_velocities('Documents/GitHub/gld-velocity-composite-10Jan19.tif')

print 'Testing read/write of composite set'
print 'Shape of composite: {}. Shape of read-in: {}.'.format(shape(v_comp), shape(vcomp_read))
print 'Testing min, max, mean value equivalence'
print nanmin(v_comp)==nanmin(vcomp_read)
print nanmax(vcomp_read)==nanmax(v_comp)
print nanmean(vcomp_read)-nanmean(v_comp) <1e-6

