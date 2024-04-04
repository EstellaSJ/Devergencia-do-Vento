# Importando bibliotecas
!pip install xarray numpy matplotlib cartopy
!pip install cartopy
!pip uninstall shapely -y
!pip install shapely --no-binary shapely
!pip install netcdf4
!pip install ncBuilder
!pip install --upgrade cartopy
!pip uninstall matplotlib -y
!pip install matplotlib==3.5.0

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from datetime import datetime
import cartopy.feature as cfeature
from ncBuilder import ncBuilder, ncHelper
import netCDF4 as nc
from scipy.io import netcdf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.colors
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates


# Carregando o arquivo .nc
dados = 'arquivo.nc'
data = xr.open_dataset(dados)
u_wind = file_path['u'][:]
v_wind = file_path['v'][:]
lons = file_path['longitude'][:]
lats = file_path['latitude'][:]
times = file_path['time'][:]
times = ncHelper.load_time(file_path.variables['time'])
idx = np.argmin(abs(times - np.array([pd.Timestamp(2023, 2, 14, 0)])))


# Definindo funções de calculo de divergência
def calculate_divergence(u, v, lon, lat):
    R_earth = 6371000.0
    dx = np.radians(lon[1] - lon[0]) * R_earth
    dy = np.radians(lat[1] - lat[0]) * R_earth
    dudx, _ = np.gradient(u, dx, axis=(1, 0))
    _, dvdy = np.gradient(v, dy, axis=(0, 1))
    divergence = (dudx + dvdy) * 100000
    return divergence

def generate_convergence_maps(data_path, output_path):
    ds = xr.open_dataset(data_path)
    lon_min, lon_max, lat_min, lat_max = -70., -10., -60, 6
    ds_region = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))

    for time in ds_region.time:
        u = ds_region.u.sel(time=time).values
        v = ds_region.v.sel(time=time).values
        lon = ds_region.longitude.values
        lat = ds_region.latitude.values
        divergence = calculate_divergence(u, v, lon, lat)

        fig_p_v, ax = plt.subplots(figsize=(20, 12), subplot_kw=dict(projection=ccrs.PlateCarree()))
        land_50m = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', linewidth=0.5, facecolor='None')
        ax.add_feature(land_50m)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', edgecolor='gray', facecolor='none', linewidth=0.5)
        ax.add_feature(states, linestyle='-', linewidth=1.2)
        countries = cfeature.BORDERS.with_scale('10m')
        ax.coastlines(resolution='10m', color='dimgray')
        ax.add_feature(countries, linestyle='-', linewidth=1.2, edgecolor='dimgray')
        cf = ax.contourf(ds_region.longitude, ds_region.latitude, divergence, cmap='RdBu', levels=np.arange(-20, 31, 5))
        strm = ax.streamplot(lon, lat, u, v, color='black', density=2, linewidth=1, arrowsize=1, transform=ccrs.PlateCarree())
        ibirite_coords = (-45.76, -23.75)
        ax.plot(ibirite_coords[0], ibirite_coords[1], 'ko', markersize=6, transform=ccrs.PlateCarree())

        formatted_date = pd.Timestamp(time.values).strftime('%Y%m%d%HZ')

        gl = ax.gridlines(draw_labels=True, dms=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = True
        gl.ylabels_left = True

        plt.title(f'Divergência do Vento (10$^{{-5}}$ s$^{{-1}}$) e \nMédia do Vento Horizontal em 250 hPa - {formatted_date}', fontsize=20, y=1.01)
        cbar_ax = fig_p_v.add_axes([0.74, 0.110, 0.020, 0.77])
        cbar = plt.colorbar(cf, cax=cbar_ax, extendrect=True, ticks=np.arange(-20, 31, 5), orientation='vertical')
        cbar.ax.yaxis.set_label_coords(1, 0.5)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='Divergência do Vento (10$^{{-5}}$ s$^{{-1}}$)', size=12, labelpad=20, position=(2.1, 0.5))

        # Ajuste do salvamento da figura
        plt.savefig(f'{output_path}/divergencia_mapa_{time}.png', bbox_inches='tight')
        plt.show()

data_path = '/local/arquivo.nc'
output_path = 'local'
generate_convergence_maps(data_path, output_path)
