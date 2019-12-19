## Making a histogram from a CSV
## 11 Feb 2019  EHU

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
#from shapely.geometry import *
from scipy import interpolate

##Read in using function from Greenland-network-troubleshooting
def read_optimization_analysis(filename, read_yieldtype=False):
    """Read a CSV file listing optimal values of yield strength for auto-selected Greenland glaciers
    Input: 
        filename
    Default arg: 
        read_yieldtype=False: determines whether we want to read and save the yield type (constant vs. Coulomb variable)
    Output: 
        Dictionary of lists including
        -Glacier ID (referenced to MEaSUREs Greenland outlets)
        -Optimal tau_y
        -Terminal bed
        -Terminal surface elevation
    """
    
    f = open(filename, 'r')
    
    header = f.readline() #header line
    #hdr = header.strip('\r\n')
    #keys = hdr.split(',') #get names of columns
    #data = {k: [] for k in keys}
    data = {'Glacier_IDs': [], #shorter keys than the names stored in CSV header
    'Optimal_taus': [],
    'Yieldtype': [],
    'Terminal_bed': [],
    'Terminal_SE': [],
    'Terminal_H': []} #adding field for ice thickness
    
    lines = f.readlines()
    f.close
    
    for i, l in enumerate(lines):
        linstrip = l.strip('\r\n')
        parts = linstrip.split(',')
        
        bed_el = float(parts[3])
        surf_el = float(parts[4])
        
        data['Glacier_IDs'].append(int(parts[0]))
        data['Optimal_taus'].append(float(parts[1]))
        if read_yieldtype: #generally won't need this
            data['Yieldtype'].append(parts[2])
        else:
            pass
        data['Terminal_bed'].append(bed_el)
        data['Terminal_SE'].append(surf_el)
        data['Terminal_H'].append(surf_el-bed_el) #calculating ice thickness from reported bed and surface elevations
    
    return data

analysis_fn = 'Documents/1. Research/2. Flowline networks/Auto_selected-networks/Optimization_analysis/bestfit_taus-B_S_smoothing-fromdate_2019-01-17.csv'
opt_data = read_optimization_analysis(analysis_fn)

cleaned_taus = opt_data['Optimal_taus']
for i, c in enumerate(cleaned_taus):
    if c==5000:
        cleaned_taus.remove(c)
    else:
        pass


## Plot histogram
tau_bins = np.arange(5, 500, 25)

#np.histogram(opt_data['Optimal_taus'], bins=tau_bins)

n, bins, patches = plt.hist(x=0.001*np.array(cleaned_taus), bins=tau_bins, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Yield strength [kPa]', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.axes().tick_params(which='both', labelsize=16)
plt.title('Optimal yield strengths of Greenland outlets')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
