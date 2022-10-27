import os
import sys
import pyproj
from osgeo import gdal
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
from osgeo import gdal, osr
import os
import matplotlib.ticker as mticker
import argparse

from corregister_wfi.raster import plot_raster as pltr


def calc_rmse(x, y):
    sum2 = x * x + y * y
    n = x.shape[0]
    sum = np.sum(sum2)
    rmse = np.sqrt(sum / n)
    return rmse

def get_args():
    parser = argparse.ArgumentParser(description = 'Plot figure 3')
    parser.add_argument('--path', help = 'path where the file are')
    parser.add_argument('--shp', help = 'shapefile')
    parser.add_argument('--rst', help = 'raster')
    parser.add_argument('--out_fig', help = 'output figure name')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    path = args.path
    shp = args.shp
    rst = args.rst
    out_fig = args.out_fig

    os.chdir(path)

    gdf = gpd.read_file(shp)
    gdf = gdf[(gdf.OUTLIER == 0)]

    x = gdf.X_SHIFT_PX
    y = gdf.Y_SHIFT_PX

    rmse = calc_rmse(x, y)

    bottom = False
    left = True
    top = True
    right = True

    ulx, uly, lrx, lry = pltr.get_extent_raster(rst)
    min_x = min(ulx, lrx)
    max_x = max(ulx, lrx)
    min_y = min(uly, lry)
    max_y = max(uly, lry)

    epsg = pltr.srs2epsg(rst)
    fuse = int(epsg[-2:])
    S = True if epsg[2] == '7' else False

    data = pltr.read_raster(rst).astype(np.float)
    data = data[::2, ::2]
    data = np.where(data == -9999, np.nan, data)

    crs = ccrs.UTM(fuse, southern_hemisphere=S)

    gdf = gpd.read_file(shp)
    gdf = gdf[(gdf.L1_OUTLIER == 0) & (gdf.L2_OUTLIER == 0)]
    vmin = gdf.ABS_SHIFT.min()
    vmax = gdf.ABS_SHIFT.max()

    gdf['srs'] = crs

    colormap = 'jet'
    fig = plt.figure(figsize=(12, 10))
    ax = pltr.make_map(fig, [0.05, 0.05, 0.9, 0.9], projection=crs, xlocs=np.arange(-90, 90, 0.5),
                  ylocs=np.arange(-90, 90, 0.5), bottom=bottom, left=left, top=top, right=right, ylabel_rotation='vertical',
                  fontsize=10)
    ax.set_extent([max_x, min_x, max_y, min_y], crs=crs)
    print('plotting...')

    ax.imshow(data, origin='upper', extent=[min_x, max_x, min_y, max_y], transform=crs, cmap='gray')

    gdf = gdf.to_crs(f"EPSG:{epsg}")

    ax.quiver(np.array(gdf.geometry.x), np.array(gdf.geometry.y), np.array(gdf.X_SHIFT_M), np.array(gdf.Y_SHIFT_M),
               np.array(gdf.ABS_SHIFT), transform=crs, color='r', cmap=colormap, clim=(vmin, vmax), units='xy', scale_units='xy')
    pltr.scale_bar(ax, length=20, location=(0.85, 0.003), height=0.01, loc='center',
                   units='km', bars=4, m_per_unit=1000, fontsize=14, zorder=12, offset=0.03)
    pltr.north_arrow(ax, 0.02, 0.0025, horizontalalignment='center', verticalalignment='bottom', path_effects=True,
                     zorder=2, fontsize=14)
    axpos = ax.get_position()
    pos_x = axpos.x0
    pos_y = axpos.y0 - 0.02
    cax_width = axpos.width
    cax_height = 0.01
    pos_cax = fig.add_axes([pos_x, pos_y, cax_width, cax_height])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=colormap)
    sm._A = []
    cbar = fig.colorbar(sm, cax=pos_cax, orientation="horizontal")
    pos_cax.text(145, -3, 'meters', fontsize=14)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(14)
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()


