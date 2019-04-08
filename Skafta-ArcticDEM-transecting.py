# Reading in ArcticDEM, sampling transect across Skafta Cauldron
# 4 Dec 2018 EHU
# Edit 21 Feb 2019 - plot analytical elastic/viscoelastic

import numpy as np
import scipy.misc as scp
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from osgeo import gdal
from netCDF4 import Dataset
from sympy.integrals.transforms import inverse_laplace_transform
from sympy import Symbol
from sympy.abc import s, t
#import shapefile
#import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
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
class Ice(object):
    """Holds constants passed to Cauldron, set to default values but adjustable.
    Default values:
        g = 9.8 m/s^2, accel due to gravity
        rho_ice = 920.0 kg m^-3, density of glacier ice
        youngmod = 3E9 Pa, Young's modulus of ice (many estimates, see e.g. 9E9 in Reeh 2003)
        poisson_nu = 0.5, Poisson's ratio for viscoelastic ice
        dyn_viscos = 1E14 Pa s, dynamic viscosity of glacier ice
        #shearmod = 3.46E9 Pa, shear modulus of glacier ice (currently set in terms of Young's modulus, Poisson's ratio)
        #lame_lambda = 5.19E9 Pa, first Lame parameter of glacier ice (currently set in terms of Young's modulus, Poisson's ratio)
    Other attributes:
        t_relax: Maxwell relaxation timescale 
    """
    def __init__(self, g=9.8, rho_ice=920.0, youngmod = 1E9, poisson_nu = 0.3, dyn_viscos = 1.5E13):
        self.g = g
        self.rho_ice = rho_ice
        self.youngmod = youngmod
        self.poisson_nu = poisson_nu
        self.dyn_viscos = dyn_viscos
        self.shearmod = self.youngmod / (2*(1+self.poisson_nu))
        self.lame_lambda = (self.youngmod * self.poisson_nu)/((1+self.poisson_nu)*(1-2*self.poisson_nu))
        self.t_relax = dyn_viscos/self.shearmod

        
class Cauldron(Ice):
    """Consistent simulation of elastic or viscoelastic cauldron collapse.
    Attributes:
        name: A string with what we call this cauldron for modelling/analysis
        thickness: ice thickness of collapsing portion in m.  Default 300 m (for Skafta)
        radius: radius of cauldron in m.  Default 1500 m (for Skafta) but should be set by observations for best match
        initial_surface: a function of radial coordinate (r=0 at center of cauldron) describing pre-collapse surface elevation
        bending_mod: (elastic) bending modulus in Pa m^3.  Calculated from other inputs.
    Inherits material properties from class Ice.
    """
    def __init__(self, name='Cauldron', thickness=300, radius=1500, initial_surface=lambda x: 1000):
        Ice.__init__(self) #inherit quantities from Ice
        self.name = name
        self.thickness = thickness
        self.radius = radius
        self.initial_surface = initial_surface
        self.bending_mod = self.youngmod * self.thickness **3 / (12*(1-self.poisson_nu**2))
    
    def elastic_beam_deform(self, x, loading=None):
        """Calculate displacement due to elastic deformation of an ice beam.  Returns deformation as a function of x, with x=0 at center of cauldron.
        Args:
            
        """
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
        
        disp = (-1*loading/(24*self.bending_mod)) * (x**4 - 2* self.radius**2 * x**2 + self.radius**4)
        
        return disp
    
    def elastic_beam_profile(self, x):
        return self.initial_surface(x) + self.elastic_beam_deform(x)
    
    def LL_radial_deform(self, r, loading=None):
        """Radially symmetric deformation according to solution presented in Landau & Lifshitz for circular plate"""
    
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
            
        LL_beta = 3 * loading *(1-self.poisson_nu**2) / (16 * self.thickness**3 * self.youngmod)
        
        LL_disp = (-1*LL_beta) * (self.radius**2 - r**2)**2
        
        return LL_disp
        
    def LL_profile(self, r):
        return self.initial_surface(r) + self.LL_radial_deform(r)
        
    def elastic_stress(self, x_eval, dx = 0.5, z=None, config='radial_plate'):
        """Calculate stress in an elastically deformed ice beam.  Returns stress at point x_eval
        Default args: 
            dx = 0.5 m, step size for finite difference approx to derivative
            z = thickness/2, distance above neutral surface to calculate stress
            config: 'beam' or 'radial_plate' (radially symmetric plate)
        """
        if z is None:
            z = 0.5 * self.thickness ##make the default location location of stress calculation the ice surface, i.e. half the ice thickness above the neutral surface
        
        if config=='beam':    
            disp_func = lambda x: self.elastic_beam_deform(x)    
            elastic_strain = z * scp.derivative(disp_func, x0=x_eval, dx=dx, n=2)
            hookean_stress =  self.youngmod * elastic_strain
            return hookean_stress
        
        if config=='radial_plate':
            disp_func = lambda x: self.LL_radial_deform(x)
            strain_rr = z * scp.derivative(disp_func, x0=x_eval, dx=dx, n=2)
            strain_thth = z * scp.derivative(disp_func, x0=x_eval, dx=dx, n=1) / self.radius
            kl_stress_rr = (self.youngmod/(1-self.poisson_nu**2)) * (strain_rr + self.poisson_nu * strain_thth) #Kirchhoff-Love stress for circular plate
            return kl_stress_rr
    
    def set_viscoelastic_bendingmod(self):
        """Construct time-dependent function viscoelastic (time-dependent) bending modulus by taking inverse Laplace transform of laplace_transformed_D.
        t0: time at which to evaluate"""
        s = Symbol('s') #Laplace variable s, to be used in SymPy computation
        t = Symbol('t', positive=True)
        laml = Symbol('laml', positive=True) #stand-in symbol for lame_lambda
        m = Symbol('m', positive=True) #stand-in symbol for shearmod
        tr = Symbol('tr', positive=True) #stand-in symbol for t_relax
        h = Symbol('h', positive=True) #stand-in symbol for thickness
        
        lambda_bar = laml + (2*m / (3*(1 + tr * s))) #transformed Lame lambda
        mu_bar = (tr * s /(1 + tr * s))*m #transformed Lame mu (shear mod)
        
        #self.lambda_bar = lambda_bar
        #self.mu_bar = mu_bar
        
        youngmod_bar = 2*mu_bar + (mu_bar*lambda_bar / (mu_bar + lambda_bar))
        poisson_bar = lambda_bar / (2*(mu_bar + lambda_bar))
        
        bending_mod_bar = youngmod_bar * h**3 / (12*(1-poisson_bar**2))
                
        symbolic_ve_D = inverse_laplace_transform(bending_mod_bar/s, s, t) #construct viscoelastic D(t) through SymPy inverse Laplace transform
        self.symbolic_ve_D = lambda t0: symbolic_ve_D.subs(((laml, self.lame_lambda), (m, self.shearmod), (tr, self.t_relax), (h, self.thickness), (t, t0)))
        #return symbolic_ve_D.subs(((t, t0), (laml, self.lame_lambda), (m, self.shearmod), (tr, self.t_relax), (h, self.thickness))) #evaluate D(t) at point t0 as expected
    
    
    def viscoelastic_deformation(self, x, t0, loading=None):
        """Collapse of a viscoelastic, radially symmetric plate solved by correspondence with elastic case."""
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
        if self.symbolic_ve_D is None:
            self.set_viscoelastic_bendingmod()
        
        ve_disp = (-1*loading/(64*self.symbolic_ve_D(t0))) * (self.radius**2 - x**2)**2 #by Laplace transform correspondence with LL_radial_deform
        
        #ve_disp = (-1*loading/(24*self.symbolic_ve_D(t0))) * (x**4 - 2* self.radius**2 * x**2 + self.radius**4)
        
        return ve_disp
    
    def viscoelastic_profile(self, x, t0):
        return self.initial_surface(x) + self.viscoelastic_deformation(x, t0)
    
    def viscoelastic_stress(self, x, t0, loading=None, z=None, config='radial_plate'):
        """Stress in a viscoelastic, radially symmetric plate by correspondence with Kirchhoff-Love elastic case"""
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
        if z is None:
            z = 0.5 * self.thickness ##make the default location location of stress calculation the ice surface, i.e. half the ice thickness above the neutral surface
        if self.symbolic_ve_D is None:
            self.set_viscoelastic_bendingmod()
        
        ve_strain_rr = (loading*z/(16*self.symbolic_ve_D(t0))) * (self.radius**2 - 3*x**2)
        ve_strain_thth = (loading*z/(16*self.symbolic_ve_D(t0))) * (self.radius**2 - x**2)
        
        ve_stress_rr = (self.youngmod / (1 - self.poisson_nu**2)) *(ve_strain_rr + self.poisson_nu * ve_strain_thth)
        
        return ve_stress_rr
    
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

crevassed_limits = (2433, 2607) #selected by inspection with ginput
w = []
for x in xaxis:
    if x>2433 and x<2607:
        w.append(1)
    else:
        w.append(0)
        
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