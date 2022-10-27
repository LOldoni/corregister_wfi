from glob import glob
import os

import math
from osgeo import gdal
import cartopy
import matplotlib as mpl

import pandas as pd
from scipy.signal import savgol_filter
import scipy.optimize as opt
from math import sqrt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from pyproj import Proj, transform
from osgeo import ogr
import sys
import matplotlib.transforms as mtrans
import datetime
from scipy.signal import argrelextrema
from matplotlib import patheffects
from concurrent.futures import ProcessPoolExecutor


_DEGREE_SYMBOL = u'\u00B0'

_MINUTE_SYMBOL = u'\u0027'

_SECOND_SYMBOL = u'\u0022'


def reverse_colourmap(cmap, name='my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def stretch_percentile(array, stretch=2, norm=True):
    if array[~np.isnan(array)].shape == (0, ):
        print("Don't possible apply stretch. All pixels are nan!")
        return array
    min = np.nanmin(array[~np.isnan(array)])
    max = np.nanmax(array[~np.isnan(array)])
    new_min = np.percentile(array[~np.isnan(array)], stretch)
    new_max = np.percentile(array[~np.isnan(array)], 100 - stretch)
    clipped_data = np.where(array < new_min, new_min, array)
    clipped_data = np.where(clipped_data > new_max, new_max, clipped_data)
    #print(min, max, new_min, new_max)
    if norm:
        out_data = clipped_data / (new_max - new_min) - new_min / (new_max - new_min)
    else:
        out_data = clipped_data * ((max - min) / (new_max - new_min))
    return out_data


def stretch_user(array, new_min, new_max, norm=True):
    clipped_data = np.where(array < new_min, new_min, array)
    clipped_data = np.where(clipped_data > new_max, new_max, clipped_data)
    #print(min, max, new_min, new_max)
    if norm:
        return clipped_data / (new_max - new_min) - new_min / (new_max - new_min)
    else:
        return clipped_data


def _transform_degrees(degrees):
    g, m, s = int(degrees), int(abs(degrees) % 1 * 60), (abs(degrees) % 1 * 60) % 1 * 60
    return g, m, s


def _fix_lons(lons):
    """
    Fix the given longitudes into the range ``[-180, 180]``.

    """
    lons = np.array(lons, copy=False, ndmin=1)
    fixed_lons = ((lons + 180) % 360) - 180
    # Make the positive 180s positive again.
    fixed_lons[(fixed_lons == -180) & (lons > 0)] *= -1
    return fixed_lons


def _lon_heimisphere(longitude):
    """Return the hemisphere (E, W or '' for 0) for the given longitude."""
    longitude = _fix_lons(longitude)
    if longitude > 0:
        hemisphere = 'E'
    elif longitude < 0:
        hemisphere = 'W'
    else:
        hemisphere = ''
    return hemisphere


def _lat_heimisphere(latitude):
    """Return the hemisphere (N, S or '' for 0) for the given latitude."""
    if latitude > 0:
        hemisphere = 'N'
    elif latitude < 0:
        hemisphere = 'S'
    else:
        hemisphere = ''
    return hemisphere


def _east_west_formatted_dms(longitude, format='g', format2='g'):
    fmt_string = u"""{lon_degree:{format}}{degree}{lon_minute:{format}}'{lon_second:{format2}}"{hemisphere}"""
    lon_degree, lon_minute, lon_second = _transform_degrees(longitude)
    return fmt_string.format(lon_degree=abs(lon_degree),
                             lon_minute=lon_minute,
                             lon_second=lon_second,
                             hemisphere=_lon_heimisphere(longitude),
                             format=format,
                             format2=format2,
                             degree=_DEGREE_SYMBOL)


def _east_west_formatted_dd(longitude, format='.2f'):
    fmt_string = u"""{lon_degree:{format}}{degree}{hemisphere}"""
    return fmt_string.format(lon_degree=abs(longitude),
                             hemisphere=_lon_heimisphere(longitude),
                             format=format,
                             degree=_DEGREE_SYMBOL)


def _north_south_formatted_dms(latitude, format='g', format2='.1f'):
    fmt_string = u"""{lat_degree:{format}}{degree}{lat_minute:{format}}'{lat_second:{format2}}"{hemisphere}"""
    lat_degree, lat_minute, lat_second = _transform_degrees(latitude)
    return fmt_string.format(lat_degree=abs(lat_degree),
                             lat_minute=lat_minute,
                             lat_second=lat_second,
                             hemisphere=_lat_heimisphere(latitude),
                             format=format,
                             format2=format2,
                             degree=_DEGREE_SYMBOL)


def _north_south_formatted_dd(latitude, format='.2f'):
    fmt_string = u"""{lat_degree:{format}}{degree}{hemisphere}"""
    return fmt_string.format(lat_degree=abs(latitude),
                             hemisphere=_lat_heimisphere(latitude),
                             format=format,
                             degree=_DEGREE_SYMBOL)


def formatter(format='dd'):
    if format == 'dd':
        #: A formatter which turns longitude values into nice longitudes such as 110W
        LONGITUDE_FORMATTER = mticker.FuncFormatter(lambda v, pos:
                                                    _east_west_formatted_dd(v))
        #: A formatter which turns longitude values into nice longitudes such as 45S
        LATITUDE_FORMATTER = mticker.FuncFormatter(lambda v, pos:
                                                   _north_south_formatted_dd(v))
    else:
        #: A formatter which turns longitude values into nice longitudes such as 110W
        LONGITUDE_FORMATTER = mticker.FuncFormatter(lambda v, pos:
                                                    _east_west_formatted_dms(v))
        #: A formatter which turns longitude values into nice longitudes such as 45S
        LATITUDE_FORMATTER = mticker.FuncFormatter(lambda v, pos:
                                                   _north_south_formatted_dms(v))
    return LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    return math.floor((lon + 180) / 6) + 1


def scale_bar(ax, length=None, location=(0.5, 0.5), height=0.02, loc='center',
              units='km', bars=4, m_per_unit=1000, fontsize=10, zorder=12, offset=0.03, patheffect_width=3):
    proj = ax.projection
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())  #

    utm = ccrs.UTM(utm_from_lon((x0 + x1) / 2))
    x0, x1, y0, y1 = ax.get_extent(utm)
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]

    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']:
                return int(x)
            else:
                return scale_number(x - 10 ** ndim)

        length = scale_number(length)
    if loc == 'center':
        bar_xs = [sbcx - length * m_per_unit / 2, sbcx + length * m_per_unit / 2]
    else:
        bar_xs = [sbcx, sbcx + length * m_per_unit]

    point1 = proj.transform_point(bar_xs[0], sbcy, utm)
    point2 = proj.transform_point(bar_xs[1], sbcy, utm)

    inv = ax.transAxes.inverted()
    p0x, p0y = inv.transform(ax.transData.transform(point1))
    p1x, p1y = inv.transform(ax.transData.transform(point2))

    width = (math.sqrt((p0x - p1x) ** 2 + (p0y - p1y) ** 2)) / bars
    partsx = np.linspace(start=p0x, stop=p0x + width * bars, num=bars + 1)
    buffer_rec = [patheffects.withStroke(linewidth=patheffect_width, foreground="w")]
    buffer_text = [patheffects.withStroke(linewidth=patheffect_width, foreground="w")]

    for i in range(0, bars):
        if i % 2 == 0:
            fill = True
            facecolor = 'k'
        else:
            fill = True
            facecolor = 'w'
        rect = mpatches.Rectangle((partsx[i], p0y),
                                  width,
                                  height,
                                  transform=ax.transAxes,
                                  fill=fill,
                                  edgecolor='k',
                                  facecolor=facecolor,
                                  zorder=zorder,
                                  clip_on=False,
                                  path_effects=buffer_rec)
        ax.add_patch(rect)
        ax.text(partsx[i], p0y + height * 1.2,
                str(round(i * (length / bars))),
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='bottom',
                color='k',
                fontsize=fontsize,
                zorder=zorder,
                path_effects=buffer_text)
    ax.text(partsx[-1], p0y + height * 1.2,
            str(round(length)),
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='bottom',
            color='k',
            fontsize=fontsize,
            zorder=zorder,
            path_effects=buffer_text)
    ax.text(partsx[-1] + offset, p0y,
            units,
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='baseline',
            color='k',
            fontsize=fontsize,
            zorder=zorder,
            path_effects=buffer_text)

    for i in range(0, bars):
        if i % 2 == 0:
            fill = True
            facecolor = 'k'
        else:
            fill = True
            facecolor = 'w'
        rect = mpatches.Rectangle((partsx[i], p0y),
                                  width,
                                  height,
                                  transform=ax.transAxes,
                                  fill=fill,
                                  edgecolor='k',
                                  facecolor=facecolor,
                                  zorder=12,
                                  clip_on=False)
        ax.add_patch(rect)


def make_map(fig, axes, projection=ccrs.PlateCarree(), fontsize=8, ylabel_rotation=False,
             xlabel_rotation=False,
             format='dd', left=True, right=True, top=True, bottom=True, xlines=True, ylines=True,
             xlocs=np.arange(-180, 180, 0.5), ylocs=np.arange(-90, 90, 0.5),
             ticks=None, outline=True, labels=True, linewidth=0.5, grid_projection=ccrs.PlateCarree()):
    ax = fig.add_axes(axes, projection=projection)

    # labels = True
    # if projection != ccrs.PlateCarree():
    #     labels = False
    gl = ax.gridlines(draw_labels=labels,
                      crs=grid_projection,
                      color='black',
                      alpha=0.5,
                      linestyle='--',
                      linewidth=linewidth,
                      zorder=5,
                      xlocs=xlocs,
                      ylocs=ylocs,
                      )
    LONGITUDE_FORMATTER, LATITUDE_FORMATTER = formatter(format=format)
    if ticks:
        ax.set_xticks(xlocs, crs=projection)
        ax.set_xticklabels([])
        ax.set_yticks(ylocs, crs=projection)
        ax.set_yticklabels([])
        ax.tick_params(top=top, right=right, bottom=bottom, left=left,
                       length=5, width=2.5, pad=0, direction='out')
        print('ticks')
    # https://stackoverflow.com/questions/58223869/cannot-remove-axis-spines-when-using-cartopy-projections-in-matplotlib
    ax.outline_patch.set_visible(outline)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlines = xlines
    gl.ylines = ylines
    gl.top_labels = top
    gl.left_labels = left
    gl.bottom_labels = bottom
    gl.right_labels = right
    gl.ylabel_style = {'fontsize': fontsize, 'rotation': ylabel_rotation, 'va': 'center'}
    gl.xlabel_style = {'fontsize': fontsize, 'rotation': xlabel_rotation, 'ha': 'center'}
    # gl.xlabel_style = {'rotation': 45}
    # ax.tick_params(direction='out', length=4, pad=1, width=1, top=True, right=True)

    return ax


class TwoPointTransformer(mtrans.Transform):
    # https://stackoverflow.com/questions/22543847/drawing-lines-between-cartopy-axes
    is_affine = False
    has_inverse = False

    def __init__(self, first_point_transform, second_point_transform):
        self.first_point_transform = first_point_transform
        self.second_point_transform = second_point_transform
        return mtrans.Transform.__init__(self)

    def transform_non_affine(self, values):
        if values.shape != (2, 2):
            raise ValueError('The TwoPointTransformer can only handle '
                             'vectors of 2 points.')
        result = self.first_point_transform.transform_affine(values)
        second_values = self.second_point_transform.transform_affine(values)
        result[1, :] = second_values[1, :]
        return result


def north_arrow(ax, x, y, horizontalalignment='center', verticalalignment='bottom', path_effects=None,
                zorder=2, fontsize=None):
    buffer=None
    if path_effects:
        buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    ax.text(x, y, u'\u25B2\nN', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment,
            path_effects=buffer, zorder=zorder, transform=ax.transAxes, fontsize=fontsize)


def read_raster(raster, ):
    if type(raster) is str:
        g = gdal.Open(raster)
    else:
        g = raster
    if g is None:
        raise IOError
    array = g.ReadAsArray()
    # return ma.array(array)
    return array


def stack_bands(filenames, block=None):
    """Returns a 3D array containing all band data from all files."""
    bands = []
    for fn in filenames:
        ds = gdal.Open(fn)
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            nodata = band.GetNoDataValue()
            if block is not None:
                data = band.ReadAsArray( \
                    block[0], block[1], block[2], block[3])
            else:
                data = band.ReadAsArray()
            data = np.where(data == nodata, np.nan, data)
            bands.append(data)
        ds = None
    return np.dstack(bands)


def stack_rasters(in_filenames, out_filename, driver='GTiff', data_type=gdal.GDT_Int16,
                nodata=-9999):
    data = stack_bands(in_filenames)
    from raster import utils as rsu
    xsize, ysize, n_bands = rsu.get_raster_info(in_filenames[0])
    geotransform = rsu.get_geotransform(in_filenames[0])
    srs =rsu.srs2wkt(in_filenames[0])
    rsu.save_raster(out_filename, data, srs, geotransform, driver=driver, gtype=data_type, setnodata=nodata,
                buildoverviews=True, computestatistics=True)


def get_extent_raster(raster):
    '''
    Get a extent a raster.
    Returns min_x, max_y, max_x, min_y
    '''
    if type(raster) is str:
        ds = gdal.Open(raster)
    else:
        ds = raster
    if ds is None:
        raise IOError
    gt = ds.GetGeoTransform()
    # inProj = Proj('+init=epsg:%s' % proj_shp, preserve_flags=True)
    ulx, uly, lrx, lry = (gt[0], gt[3], gt[0] + gt[1] * ds.RasterXSize, \
                          gt[3] + gt[5] * ds.RasterYSize)
    return ulx, uly, lrx, lry


# def convert_coords(x, y, in_epsg, out_epsg):
#     inProj = Proj('epsg:{}'.format(in_epsg))
#     outProj = Proj('epsg:{}'.format(out_epsg))
#     return transform

def convert_coords(x, y, in_proj, out_proj):
    from pyproj import Transformer
    # inProj = Proj(in_proj)
    # outProj = Proj(out_proj)
    # return transform(inProj, outProj, x, y)
    transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)
    return transformer.transform(x, y)


def stack_bands(filenames, block=None):
    """Returns a 3D array containing all band data from all files."""
    bands = []
    for fn in filenames:
        ds = gdal.Open(fn)
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            nodata = band.GetNoDataValue()
            if block is not None:
                data = band.ReadAsArray( \
                    block[0], block[1], block[2], block[3])
            else:
                data = band.ReadAsArray()
            data = np.where(data == nodata, np.nan, data)
            bands.append(data)
        ds = None
    return np.dstack(bands)


def get_XoffYoff(raster, x, y):
    if type(raster) == str:
        in_ds = gdal.Open(raster)
    else: in_ds = raster
    in_gt = in_ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(in_gt)
    offsets = gdal.ApplyGeoTransform(inv_gt, x, y)
    xoff, yoff = map(int, offsets)
    del in_ds
    return xoff, yoff

def get_xy(raster, xoff, yoff):
    if type(raster) == str:
        in_ds = gdal.Open(raster)
    else: in_ds = raster
    in_gt = in_ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(in_gt)
    offsets = gdal.ApplyGeoTransform(in_gt, xoff, yoff)
    x, y = map(int, offsets)
    del in_ds
    return x, y

def get_geotransform(raster):
    """

    :rtype: object
    """
    if type(raster) is str:
        in_ds = gdal.Open(raster)
    else:
        in_ds = raster
    if in_ds is None:
        raise IOError
    in_gt = in_ds.GetGeoTransform()
    del in_ds
    return in_gt


def srs2proj4(raster):
    """

    :rtype: SRS as proj4 format
    """
    from osgeo import osr
    if type(raster) is str:
        g = gdal.Open(raster)
    else:
        g = raster
    if g is None:
        raise IOError
    proj = osr.SpatialReference(wkt=g.GetProjection())
    return proj.ExportToProj4()

def srs2epsg(raster):
    from osgeo import osr
    if type(raster) is str:
        g = gdal.Open(raster)
    else:
        g = raster
    if g is None:
        raise IOError
    proj = osr.SpatialReference(wkt=g.GetProjection())
    return proj.GetAttrValue('AUTHORITY', 1)

def srs2wkt(raster):
    from osgeo import osr
    from osgeo import gdal
    if type(raster) is str:
        g = gdal.Open(raster)
    else:
        g = raster
    if g is None:
        raise IOError
    proj = osr.SpatialReference(wkt=g.GetProjection())
    return proj.ExportToWkt()


def read_raster_bbox(raster, extent):
    xmin, xmax, ymin, ymax = extent
    if type(raster) == str:
        in_ds = gdal.Open(raster)
    else:
        in_ds = raster
    xsize = in_ds.RasterXSize
    ysize = in_ds.RasterYSize
    bands = in_ds.RasterCount
    in_gt = in_ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(in_gt)
    offsets_ul = gdal.ApplyGeoTransform(inv_gt, xmin, ymax)
    off_ulx, off_uly = map(int, offsets_ul)
    offsets_lr = gdal.ApplyGeoTransform(inv_gt, xmax, ymin)
    off_lrx, off_lry = map(int, offsets_lr)

    off_lry = min(off_lry, ysize)
    off_lrx = min(off_lrx, xsize)
    off_lry = max(off_lry, 0)
    off_lrx = max(off_lrx, 0)

    off_uly = max(off_uly, 0)
    off_ulx = max(off_ulx, 0)
    off_uly = min(off_uly, ysize)
    off_ulx = min(off_ulx, xsize)

    rows = off_lry - off_uly
    cols = off_lrx - off_ulx
    if cols == 0 or rows == 0:
        in_ds = None
        print(f'window has cols = {cols} and rows = {rows}')
        return None
    if bands == 1:
        band = in_ds.GetRasterBand(1)
        data = band.ReadAsArray(off_ulx, off_uly, cols, rows)
    else:
        data = []
        for i in range(bands):
            band = in_ds.GetRasterBand(i + 1)
            data.append(band.ReadAsArray(off_ulx, off_uly, cols, rows))
        data = np.dstack(data)
    in_ds = None
    return data

def calc_add_axes(cols, rows, hs, vs, hsb, vsb):
    axes = []
    height = (1 - 2 * hs - (rows - 1) * hsb) / rows
    width = (1 - 2 * vs - (cols - 1) * vsb) / cols
    for i in range(rows):
        for j in range(cols):
            if i == 0:
                y = hs
            else:
                y = hs + (hsb + height) * i
            if j == 0:
                x = vs
            else:
                x = vs + (vsb + width) * j
            axes.append([x, y, width, height])
    return axes

def calc_add_axes2(cols, rows, horizontal_space_left, horizontal_space_right, vertical_space_top,
                  vertical_space_bottom, vertical_space, horizontal_space):
    axes = []
    height = (1 - (vertical_space_bottom + vertical_space_top) - (rows - 1) * vertical_space) / rows
    # width = (1 - (horizontal_space_left + horizontal_space_right) - (cols - 1) * horizontal_space) / cols

    width = (1 - horizontal_space_left - horizontal_space_right - (cols-1) * horizontal_space)/ cols
    for i in range(rows):
        for j in range(cols):
            if i == 0:
                y = vertical_space_bottom
            else:
                y = vertical_space_bottom + (horizontal_space + height) * i
            if j == 0:
                x = horizontal_space_left
            else:
                x = horizontal_space_left + (vertical_space + width) * j
            axes.append([x, y, width, height])
    return axes


def read_rasters_ts(rasters, x, y):
    ts = []
    for raster in rasters:
        xoff, yoff = get_XoffYoff(raster, x, y)
        in_ds = gdal.Open(raster)
        band = in_ds.GetRasterBand(1)
        data = band.ReadAsArray(xoff, yoff, 1, 1)[0, 0]
        ts.append(data)
    return ts

def get_pixel_corner(raster, xoff, yoff):
    '''
    Get a extent a raster.
    Returns min_x, max_y, max_x, min_y
    '''
    if type(raster) is str:
        ds = gdal.Open(raster)
    else:
        ds = raster
    if ds is None:
        raise IOError
    gt = ds.GetGeoTransform()
    # inProj = Proj('+init=epsg:%s' % proj_shp, preserve_flags=True)
    ulx, uly, lrx, lry = gt[0] + gt[1] * xoff, gt[3] + gt[5] * yoff, gt[0] + gt[1] * (xoff + 1), gt[3] + gt[5] * (yoff + 1)
    return ulx, uly, lrx, lry