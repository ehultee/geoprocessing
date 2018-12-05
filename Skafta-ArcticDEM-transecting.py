# Reading in ArcticDEM, sampling transect across Skafta Cauldron
# 4 Dec 2018 EHU

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from osgeo import gdal
from netCDF4 import Dataset
import shapefile
import datetime
import matplotlib.pyplot as plt
import math
#from matplotlib.colors import LogNorm
#from matplotlib import cm
##from shapely.geometry import *
#from scipy import interpolate
#from scipy.ndimage import gaussian_filter


## Functions to read in surface from ArcticDEM - as GeoTIFF and NetCDF
def read_ArcticDEM_tif(filename, return_grid=True, return_proj=False):
    """Extract x, y, v from an ArcticDEM GeoTIFF"""
    ds = gdal.Open(filename)
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    
    geotransform = ds.GetGeoTransform()
    xOrigin = geotransform[0]
    xPix = geotransform[1] #pixel width in x-direction
    yOrigin = geotransform[3]
    yPix = geotransform[5] #pixel width in y-direction
    
    lons = xOrigin + np.arange(0, ncols)*xPix
    lats = yOrigin + np.arange(0, nrows)*yPix
    
    x, y = np.meshgrid(lons, lats)
    
    hband = ds.GetRasterBand(1)
    harr = hband.ReadAsArray()
    
    if return_grid and return_proj:
        return x, y, harr, ds.GetProjection()
    elif return_grid:
        return x, y, harr
    else:
        return harr
    

def read_ArcticDEM_nc(filename, return_grid=True):
    fh = Dataset(filename, mode='r')
    lon = fh.variables['lon'][:].copy() #longitude
    lat = fh.variables['lat'][:].copy() #latitude
    se = fh.variables['Band1'][:].copy() #assuming the variable called "GDAL Band Number 1" is actually surface elevation
    
    if return_grid:
        return lon, lat, se
    else:
        return se

skafta_region_path = 'Documents/6. MIT/Skaftar collapse/data/arcticDEM/'
nc_20121015_path = skafta_region_path + 'subset_nc/SETSM_WV02_20121015_skaftar_east_ll.nc'
nc_20151010_path = skafta_region_path + 'subset_nc/SETSM_WV02_20151010_skaftar_east_ll.nc'

lon_2012, lat_2012, se_2012 = read_ArcticDEM_nc(nc_20121015_path)
SE_2012 = np.ma.masked_where(se_2012==0, se_2012)
lon_2015, lat_2015, se_2015 = read_ArcticDEM_nc(nc_20151010_path)
SE_2015 = np.ma.masked_where(se_2015==0, se_2015)

## Interpolating surface elevation and sampling transect
sefunc_2012 = interpolate.interp2d(lon_2012, lat_2012, SE_2012)
sefunc_2015 = interpolate.interp2d(lon_2015, lat_2015, SE_2015)

npoints = 1000
endpoints = [(-17.542113802658239, 64.488141277357315),
 (-17.48586677277758, 64.486397775690023)]
lonvals = np.linspace(endpoints[0][0], endpoints[1][0], npoints)
latvals = np.linspace(endpoints[0][1], endpoints[1][1], npoints)
sevals_2012 = np.asarray([sefunc_2012(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()
sevals_2015 = np.asarray([sefunc_2015(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()


## Plotting transect, with x-axis of distance along transect in m
def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

transect_length = haversine(endpoints[0], endpoints[1])
xaxis = linspace(0, transect_length, num=npoints)

plt.figure()
plt.plot(xaxis, sevals_2012, color='k', ls='-.', label='15 Oct 2012')
plt.plot(xaxis, sevals_2015, color='k', ls='-', label='10 Oct 2015')
plt.fill_between(xaxis, sevals_2012, sevals_2015, color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
plt.fill_between(xaxis, sevals_2015, (plt.axes().get_ylim()[0]), color='Azure')
plt.axes().set_aspect(5)
plt.axes().set_xlim(0, transect_length)
plt.axes().set_yticks([1550, 1600, 1650, 1700])
plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
plt.show()
plt.savefig('Skafta-transect-aspect_5.png', transparent=True)