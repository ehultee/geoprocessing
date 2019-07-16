# Reading in ArcticDEM, sampling transect across Skafta Cauldron
# 4 Dec 2018 EHU
# Edit 21 Feb 2019 - plot analytical elastic/viscoelastic
# Edit 16 July - move functions to helper module

import numpy as np
import scipy.misc as scp
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from osgeo import gdal
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import math


## Read in ArcticDEM surface
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
 (-17.48586677277758, 64.486397775690023)] #coordinates at either side of the cauldron, selected by inspection with ginput.  
lonvals = np.linspace(endpoints[0][0], endpoints[1][0], npoints)
latvals = np.linspace(endpoints[0][1], endpoints[1][1], npoints)
sevals_2012 = np.asarray([sefunc_2012(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()
sevals_2015 = np.asarray([sefunc_2015(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()

## Prepare transect for plotting, with x-axis of distance along transect in m
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

transect_length = haversine(endpoints[0][::-1], endpoints[1][::-1])
xaxis = np.linspace(0, transect_length, num=npoints)



## Set up analytical profile    
x_cylcoords = np.linspace(-0.5*transect_length, 0.5*transect_length, num=npoints)
initial_surf_val = np.mean((sevals_2012[0], sevals_2012[-1])) #surface elevation at edges before loading
initial_surf = interpolate.interp1d(x_cylcoords, sevals_2012, kind='quadratic') #will have matching length as long as num=npoints in x_cylcoords above
ESkafta = Cauldron(name='Eastern_Skafta', initial_surface = initial_surf, radius = 0.5*transect_length)
ESkafta.set_viscoelastic_bendingmod()

stress_array = [ESkafta.elastic_stress(x) for x in x_cylcoords]
elas_profile_array = [ESkafta.elastic_beam_profile(x) for x in x_cylcoords]
LL_profile_array = [ESkafta.LL_profile(x) for x in x_cylcoords]
nseconds = 5*24*60*60 #number of seconds in the roughly 5-day collapse period
times = np.arange(0, nseconds, step=20000)
ve_profile_series = [[ESkafta.viscoelastic_profile(x, t0) for x in x_cylcoords] for t0 in times]

elas_beam_stress = [ESkafta.elastic_stress(x, config='beam') for x in x_cylcoords]
elas_plate_stress = [ESkafta.elastic_stress(x, config='radial_plate') for x in x_cylcoords]
ve_plate_stress_min = [ESkafta.viscoelastic_stress(x, times[0]) for x in x_cylcoords]
ve_plate_stress_max = [ESkafta.viscoelastic_stress(x, times[4]) for x in x_cylcoords]
ve_bendingmod_series = [ESkafta.symbolic_ve_D(t0) for t0 in times]

#crevassed_limits = (2433, 2607) #selected by inspection with ginput
#w = []
#for x in xaxis:
#    if x>2433 and x<2607:
#        w.append(1)
#    else:
#        w.append(0)
        
## Make figure

cmap = cm.get_cmap('winter_r')
#colors = cmap([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
colors = cmap(np.linspace(0.1, 0.9, num=len(times)+1))

plt.figure('Elastic only', figsize=(7, 3))
plt.plot(xaxis, sevals_2012, color='k', ls='-.') #, label='15 Oct 2012'
plt.plot(xaxis, sevals_2015, color='k', ls='-', label='Obs.') #, label='10 Oct 2015'
plt.plot(xaxis, elas_profile_array, color='r', ls=':', label='Elastic beam')
plt.plot(xaxis, LL_profile_array, color=colors[0], lw=2, label='Elastic plate')
plt.fill_between(xaxis, sevals_2012, sevals_2015, color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
plt.fill_between(xaxis, sevals_2015, (plt.axes().get_ylim()[0]), color='Azure')
plt.legend(loc='lower left')
plt.axes().set_aspect(5)
plt.axes().set_xlim(0, transect_length)
plt.axes().set_yticks([1550, 1600, 1650, 1700])
#plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
plt.axes().tick_params(which='both', labelsize=14)
#plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
plt.axes().set_xlabel('Along-transect distance [m]', fontsize=16)
plt.axes().set_ylabel('Surface elevation [m a.s.l.]', fontsize=16)
#plt.title('Eastern Skafta cauldron transect: observed, ideal elastic, ideal viscoelastic. E={:.1E}'.format(ESkafta.youngmod), fontsize=18)
plt.show()
#plt.savefig('Skafta-transect-aspect_5.png', transparent=True)

plt.figure('Viscoelastic progression', figsize=(7, 3))
plt.plot(xaxis, sevals_2012, color='k', ls='-.') #, label='15 Oct 2012'
plt.plot(xaxis, sevals_2015, color='k', ls='-', label='Obs.') #, label='10 Oct 2015'
#plt.plot(xaxis, elas_profile_array, color='r', ls=':', label='Elastic beam')
plt.plot(xaxis, LL_profile_array, color=colors[0], lw=2, label='Elastic plate')
for i,ti in enumerate(times[::10]):
    labeltime = int(round(ti/86400)) #time in days
    plt.plot(xaxis, ve_profile_series[i][:], ls='--', color=colors[i+1], lw=2, label='Viscoelastic, t = {} days'.format(labeltime))
plt.fill_between(xaxis, sevals_2012, sevals_2015, color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
plt.fill_between(xaxis, sevals_2015, (plt.axes().get_ylim()[0]), color='Azure')
plt.legend(loc='lower left')
plt.axes().set_aspect(5)
plt.axes().set_xlim(0, transect_length)
plt.axes().set_yticks([1550, 1600, 1650, 1700])
#plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
plt.axes().tick_params(which='both', labelsize=14)
#plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
plt.axes().set_xlabel('Along-transect distance [m]', fontsize=16)
plt.axes().set_ylabel('Surface elevation [m a.s.l.]', fontsize=16)
#plt.title('Eastern Skafta cauldron transect: observed, ideal elastic, ideal viscoelastic. E={:.1E}, eta={:.1E}'.format(ESkafta.youngmod, ESkafta.dyn_viscos), fontsize=18)
plt.show()
#plt.savefig('Desktop/Skafta-viscoelastic-progression-{}.png'.format(datetime.date.today()), transparent=True)
#
#plt.figure('Elastic stress')
#plt.plot(xaxis, 1E-6*np.array(elas_plate_stress), color='k', ls='-', lw=2, label='Elastic plate')
#plt.plot(xaxis, np.zeros(len(xaxis)), color='b', ls=':')
##plt.plot(xaxis, 1E-6*np.array(elas_beam_stress), color='k', ls='-.', label='Elastic beam')
#plt.fill_between(xaxis, 1E-6*np.array(elas_plate_stress), plt.axes().get_ylim()[0], where=w, color='r', alpha=0.5)
##plt.axes().add_patch(Rectangle((2433.5,plt.axes().get_ylim()[0]), 2607-2433.5, plt.axes().get_ylim()[1]-plt.axes().get_ylim()[0], facecolor='r', alpha=0.5))
#plt.legend(loc='upper right')
#plt.axes().tick_params(which='both', labelsize=14)
#plt.axes().set_xlim(0, transect_length)
#plt.axes().set_xlabel('Along-transect distance [m]', fontsize=16)
#plt.axes().set_ylabel('Elastic stress [MPa]', fontsize=16)
#plt.title('Elastic stress at cauldron surface', fontsize=18)
#plt.show()
#
#plt.figure('Stress comparison')
#plt.plot(xaxis, 1E-6*np.array(elas_plate_stress), color='k', ls='-', lw=2, label='Elastic plate')
#plt.plot(xaxis, 1E-6*np.array(ve_plate_stress_max), color='k', ls='-.', lw=2, label='Viscoelastic plate, t={}'.format(max(times)))
#plt.plot(xaxis, np.zeros(len(xaxis)), color='b', ls=':')
##plt.plot(xaxis, 1E-6*np.array(elas_beam_stress), color='k', ls='-.', label='Elastic beam')
#plt.fill_between(xaxis, 1E-6*np.array(elas_plate_stress), plt.axes().get_ylim()[0], where=w, color='r', alpha=0.5)
##plt.axes().add_patch(Rectangle((2433.5,plt.axes().get_ylim()[0]), 2607-2433.5, plt.axes().get_ylim()[1]-plt.axes().get_ylim()[0], facecolor='r', alpha=0.5))
#plt.legend(loc='upper right')
#plt.axes().tick_params(which='both', labelsize=14)
#plt.axes().set_xlim(0, transect_length)
#plt.axes().set_xlabel('Along-transect distance [m]', fontsize=16)
#plt.axes().set_ylabel('Elastic stress [MPa]', fontsize=16)
#plt.title('Stress at cauldron surface', fontsize=18)
#plt.show()