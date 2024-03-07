import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.shapereader as sr

from matplotlib.colors import LinearSegmentedColormap

purples = LinearSegmentedColormap.from_list('purples', plt.cm.get_cmap("Purples")(np.arange(0, 1.1, .1)), N = 10)
ylorrd = LinearSegmentedColormap.from_list('ylorrd', plt.cm.get_cmap("YlOrRd")(np.arange(0, 1.1, .1)), N = 7)


def set_plt_rc_params():
    plt.rcParams['figure.dpi'] = 120
    plt.rc('axes', titlesize=8)     # fontsize of the axes titles (i.e. title of each panel)
    plt.rc('axes', labelsize=7)    # fontsize of the x and y axis labels
    plt.rc('xtick', labelsize=7)    # fontsize of the x tick labels
    plt.rc('ytick', labelsize=7)    # fontsize of the y tick labels
    plt.rc('figure', titlesize = 8)
    plt.rc('legend', fontsize=7)
    plt.rc('legend', title_fontsize=7)
    plt.rc('lines', linewidth=1)
    

def plot_map(dat, lons, lats, ax, CMAP = None, VMIN = None, VMAX = None, na_col = "whitesmoke", 
            label = None, legend = True):
    """ """
    if CMAP is None: 
        CMAP = "viridis"
    
    if na_col is not None:
        ax.set_facecolor(na_col)
    else: 
        ax.set_facecolor(plt.get_cmap(CMAP)(0))
    
    if VMIN is not None:
        p = ax.pcolormesh(lons, lats, dat, vmin = VMIN, vmax = VMAX, transform = ccrs.PlateCarree(), 
                          cmap = CMAP)
    else: 
        p = ax.pcolormesh(lons, lats, dat, transform = ccrs.PlateCarree(), cmap = CMAP)
    
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor = 'lightgray')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.5)   
    if (label is not None) and (legend): 
        plt.colorbar(p, ax = ax, shrink = 0.7, location = "bottom", label = label)
    elif legend: 
        plt.colorbar(p, ax = ax, shrink = 0.7, location = "bottom")

    return(p)

def add_regions(ax):
    ax.add_geometries(sr.Reader("../processed_data/region_shapefiles/tropical_pacific.shp").geometries(), 
                  crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = "black", linewidth = 0.5)

    ax.add_geometries(sr.Reader("../processed_data/region_shapefiles/tropical_atlantic_260.shp").geometries(), 
                  crs = ccrs.PlateCarree(central_longitude = 260), facecolor = 'none', edgecolor = "black", linewidth = 0.5)

    ax.add_geometries(sr.Reader("../processed_data/region_shapefiles/north_pacific.shp").geometries(), 
                  crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = "black", linewidth = 0.5)

    ax.add_geometries(sr.Reader("../processed_data/region_shapefiles/north_atlantic_260.shp").geometries(), 
                   crs = ccrs.PlateCarree(central_longitude = 260), facecolor = 'none', edgecolor = "black", linewidth = 0.5)

    ax.add_geometries(sr.Reader("../processed_data/region_shapefiles/southern_ocean_260.shp").geometries(), 
                  crs = ccrs.PlateCarree(central_longitude = 260), facecolor = 'none', edgecolor = "black", linewidth = 0.5)

    ax.add_geometries(sr.Reader("../processed_data/region_shapefiles/west_indian.shp").geometries(), 
                  crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = "black", linewidth = 0.5)