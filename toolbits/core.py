
from __future__ import absolute_import, print_function, division

import os
import warnings

import numpy as np
import netCDF4
from loguru import logger


logger.disable(__name__)


def warningfilter(action, category=RuntimeWarning):
    def warning_deco(func):
        def func_wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(action, category)
                return func(*args, **kwargs)
        return func_wrapper
    return warning_deco


def linear_wls(xi, yi, wi=None, axis=0):
    """
    Ordinary least squares fitting with weights
    """
    
    if wi is None:
        wi = np.ones_like(yi)

    def sum(ar):
        return np.nansum(ar, axis=axis)

    wi2 = wi * wi
    slope = (sum(wi2) * sum(wi2 * xi * yi) - sum(wi2 * yi) * sum(wi2 * xi)) / \
        (sum(wi2) * sum(wi2 * xi * xi) - sum(wi2 * xi)**2)
    intercept = (sum(wi2 * yi) - slope * sum(wi2 * xi)) / sum(wi2)

    return slope, intercept


def regrid(grid_x, grid_y, grid_z, x, y, method='nearest'):
    """Interpolation along axes (-2,-1) in rank-n grid_z.

    Parameters
    ----------
    grid_x: array-like, rank 1 of shape (N,)
        coordinate values along dimension x of grid_z (axis=-1)
    grid_y: array-like, rank 1 of shape (M,)
        coordinate values along dimension y of grid_z (axis=-2)
    grid_z: array-like, rank n of shape (..., M, N)
    x: array-like, arbitrary shape, typically rank-1 or rank-2 array
        target coordinate values for dimension x. Must have same shape as y
    y: array-like, arbitrary shape, typically rank-1 or rank-2 array
        target coordinate values for dimension y. Must have same shape as x
    method: str
        interpolation method: nearest or bilinear

    Return
    ------
    Interpolated values in an array with shape (..., shape of x and y). For
    instance, if grid_z has dimensions (dfb, slot, latitude, longitude) and x
    and y are rank-2 longitude and latitude arrays, respectively, with shape
    (P, Q), the output array would have shape (dfb, slot, P, Q). In contrast,
    if the new locations were rank-1 arrays with shape (R,), the shape
    of the output array would be (dfb, slot, R). Same comments apply to input
    arrays with shapes (time, latitude, longitude) or (issue_day, cycle,
    lead_hour, latitude, longitude), for instance.
    """
    # transformation to the segment (0,1)x(0,1)
    def normalize(v, grid):
        return (v - grid[0]) / (grid[-1] - grid[0])
    ycoords = normalize(grid_y, grid_y)
    xcoords = normalize(grid_x, grid_x)
    yinterp = normalize(y, grid_y)
    xinterp = normalize(x, grid_x)

    zvalues = grid_z
    if np.ma.is_masked(zvalues):
        zvalues = np.where(zvalues.mask, np.nan, zvalues.data)
    assert zvalues.ndim >= 2, \
        'grid_val must have at least ndim=2. Got {}'.format(zvalues.ndim)

    def clip(k, kmax):
        return np.clip(k, 0, kmax)

    if method == 'nearest':
        jx = np.rint((grid_y.size - 1) * yinterp).astype(np.int)
        ix = np.rint((grid_x.size - 1) * xinterp).astype(np.int)
        jx = clip(jx, grid_y.size - 1)
        ix = clip(ix, grid_x.size - 1)
        return zvalues[..., jx, ix]

    elif method == 'bilinear':
        j1 = ((grid_y.size - 1) * yinterp).astype(np.int)
        i1 = ((grid_x.size - 1) * xinterp).astype(np.int)
        jmax, imax = grid_y.size - 1, grid_x.size - 1
        Axy = (ycoords[clip(j1 + 1, jmax)] - ycoords[clip(j1, jmax)]) * \
            (xcoords[clip(i1 + 1, imax)] - xcoords[clip(i1, imax)])
        A11 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
            (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
        A12 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
            (xinterp - xcoords[clip(i1, imax)]) / Axy
        A21 = (yinterp - ycoords[clip(j1, jmax)]) * \
            (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
        A22 = (yinterp - ycoords[clip(j1, jmax)]) * \
            (xinterp - xcoords[clip(i1, imax)]) / Axy
        return (zvalues[..., clip(j1, jmax), clip(i1, imax)] * A11 +
                zvalues[..., clip(j1, jmax), clip(i1 + 1, imax)] * A12 +
                zvalues[..., clip(j1 + 1, jmax), clip(i1, imax)] * A21 +
                zvalues[..., clip(j1 + 1, jmax), clip(i1 + 1, imax)] * A22)

    else:
        raise ValueError('unknown interpolation method %r' % method)


def num2date(*args, **kwargs):
    return netCDF4.num2date(*args, **kwargs)


def date2num(*args, **kwargs):
    return netCDF4.date2num(*args, **kwargs)


def write_netcdf(out_fn, variables, global_attributes={}, var_options={},
                 format='NETCDF4_CLASSIC'):
    """A minimual dummy use example.

    from jararias_toolbox import write_netcdf
    import netCDF4
    from datetime import datetime
    the_time = datetime(2010, 1, 1, 12, 0)
    time_units = 'hours since 1970-01-01'
    variables = {
        'time': {
            'dimensions': ('time',),
            'values': netCDF4.date2num(the_time, units=time_units),
            'attributes': {
                'description': 'time description',
                'units': time_units}
            },
        'south_north': {
            'dimensions': ('south_north',),
            'values': np.arange(40),
            'attributes': {
                'description': 'south_north description',
                'units': 'south_north units'}
            },
        'east_west': {
            'dimensions': ('east_west',),
            'values': np.arange(60),
            'attributes': {
                'description': 'east_west description',
                'units': 'east_west units'}
            },
        'random_values': {
            'dimensions': ('time', 'south_north', 'east_west'),
            'values': np.random.randn(1, 40, 60),
            'options': {
                'fill_value': 123,
                'least_significant_digit': 3},
            'attributes': {
                'description': 'values description',
                'units': 'values units'}
            }
    }
    write_netcdf('aname.nc4', variables, var_options={'zlib': True})
    """
    var_options.setdefault('datatype', 'f4')
    var_options.setdefault('zlib', True)

    kwargs = dict(mode='w', format=format)
    with netCDF4.Dataset(out_fn, **kwargs) as cdf:

        for varname, variable in variables.items():

            dimensions = variable['dimensions']

            # first, check that all required dimensions have
            # been already pushed in the netcdf file
            for dim_i, dim_name in enumerate(dimensions):

                if dim_name in cdf.dimensions:
                    continue

                if dim_name not in variables:
                    # raise ValueError('missing dimension variable {}'
                    #                  .format(dim_name))
                    len_dim = variables[varname]['values'].shape[dim_i]
                    cdf.createDimension(dim_name, len_dim)
                    continue

                # if dim_value is an scalar, an UNLIMITED dimension is created
                dim_values = variables[dim_name]['values']
                variables[dim_name].setdefault('unlimited', False)
                if variables[dim_name]['unlimited'] is True:
                    cdf.createDimension(dim_name, None)
                else:
                    try:
                        len_dim = len(dim_values)
                    except Exception:
                        len_dim = 1
                    cdf.createDimension(dim_name, len_dim)
                cdf.sync()

                opts = dict(var_options)
                if 'options' in variables[dim_name]:
                    opts.update(variables[dim_name].pop('options'))
                opts['dimensions'] = (dim_name,)

                var = cdf.createVariable(dim_name, **opts)
                var[:] = dim_values

                if 'attributes' in variables[dim_name]:
                    attributes = variables[dim_name]['attributes']
                    for attr_name, attr_value in attributes.items():
                        setattr(var, attr_name, attr_value)

                cdf.sync()

            if varname in cdf.variables:
                continue

            # then, push the variable
            opts = dict(var_options)
            if 'options' in variable:
                opts.update(variable.pop('options'))

            values = variable['values']
            if np.ma.isMaskedArray(values) or np.any(np.isnan(values)):
                fill_value = opts.setdefault('fill_value', np.nan)
                if fill_value is None:
                    fill_value = np.nan
                    opts['fill_value'] = fill_value
                if np.ma.isMaskedArray(values):
                    values = np.where(values.mask, fill_value, values.data)
                else:
                    values = np.where(np.isnan(values), fill_value, values)
            opts['dimensions'] = dimensions

            var = cdf.createVariable(varname, **opts)
            var[:] = values

            if 'attributes' in variable:
                for attr_name, attr_value in variable['attributes'].items():
                    if attr_name == '_FillValue':
                        continue
                    setattr(var, attr_name, attr_value)

            cdf.sync()

        # global attributes...
        for attr_name, attr_value in global_attributes.items():
            setattr(cdf, attr_name, attr_value)


@warningfilter('ignore', UserWarning)
def read_netcdf(fn_or_fns, skip_variables=None):

    dispatcher = netCDF4.Dataset if isinstance(fn_or_fns, str) else netCDF4.MFDataset

    cdf_variables = {}
    cdf_attributes = {}

    with dispatcher(fn_or_fns, mode='r') as cdf:

        # so that dimensions come before variables in the dictionary...
        dimensions = list(cdf.dimensions.keys())
        variables = list(set(cdf.variables.keys()).difference(dimensions))

        if skip_variables is not None:
            if set(variables).intersection(skip_variables):
                variables = list(set(variables).difference(skip_variables))

        for var_name in dimensions + variables:
            logger.debug(f'Reading {var_name}')
            var_dict = {}
            var_obj = cdf.variables.get(var_name, None)
            if var_obj is None and var_name in dimensions:
                continue
            var_dict['dimensions'] = var_obj.dimensions
            var_dict['values'] = var_obj[:]
            var_dict['attributes'] = {}
            if var_name in dimensions:
                var_dict['unlimited'] = cdf.dimensions[var_name].isunlimited()
            for attr_name in var_obj.ncattrs():
                var_dict['attributes'][attr_name] = getattr(var_obj, attr_name)
            cdf_variables.update({var_name: var_dict})

        for attr_name in cdf.ncattrs():
            cdf_attributes[attr_name] = getattr(cdf, attr_name)

    return cdf_variables, cdf_attributes


# if __name__ == '__main__':

#     from datetime import datetime
#     import sys

#     fn = ('/media/Toshiba_3TB/data/aerosol/merra2/hourly/'
#           '2018/merra2_aer_ext_550_hourly_20180101.nc4')

#     variables, attributes = read_netcdf(fn)

#     # change time unit reference. In the original format, the time
#     # reference for units changes from one file to another. I want
#     # to have the same time reference for all files
#     t0 = datetime(1980, 1, 1, 0, 0)
#     time_units = f'hours since {t0}'
#     old_times = netCDF4.num2date(
#         variables['time']['values'],
#         units=variables['time']['attributes']['units'])
#     new_times = netCDF4.num2date(
#         [(t - t0).total_seconds() / 3600. for t in old_times],
#         units=time_units)
#     new_times = [(t - t0).total_seconds() / 3600. for t in old_times]
#     variables['time']['values'] = new_times
#     variables['time']['attributes']['units'] = time_units

#     variables['TOTEXTTAU'].update({
#         'options': {'least_significant_digit': 4}})

#     write_netcdf('kk.nc4', variables=variables, global_attributes=attributes,
#                  var_options={'zlib': True})

#     variables, attributes = read_netcdf('kk.nc4')
#     print(variables.keys())
