import numpy as np
import struct
import pyart
import pandas as pd
import os
from tqdm import tqdm

def scn_to_sweep(fn, position = None):
    with open(fn, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    progress = struct.unpack('H',fileContent[:2])[0]
    header_info  = struct.unpack('=HHHHHHHHhHHhHHHHHHHhhHHHlhlhHHHHHHHHHH',fileContent[:progress])
    nrays, ngates = header_info[21], header_info[22]
    dr = header_info[23]/100
    rpm = header_info[16] / 10
    if position is None:
        rlat = (-1 if header_info[8] < 0 else 1) * (abs(header_info[8]) + header_info[9] / 60 + header_info[10] / (1000 * 60 * 60))
        rlon = (-1 if header_info[11] < 0 else 1) * (abs(header_info[11]) + header_info[12] / 60 + header_info[13] / (1000 * 60 * 60))
        position = (rlat, rlon)
    nfields = 9
    sf = np.array([[1, 32768,100,0]]*5 + [[360,32768,65535,0]] + [[2,1,65534,0]] + [[1,1,100,0]] + [[1,0,1,-1]])
    out, az, el = [], [], []
    for i in range(nrays):
        antenna_info = struct.unpack('=HHhH',fileContent[progress:progress+2*4])
        az.append(antenna_info[1]/100)
        el.append(antenna_info[2]/100)
        
        progress += 2*4
        
        tgates = ngates * nfields
        raw_data = struct.unpack('='+'H'*tgates,fileContent[progress:progress+2*tgates])
        out.append(np.array(raw_data).reshape(ngates, nfields, order = 'F'))
        progress += 2*tgates
    
    out, az, el = np.array(out), np.array(az), np.array(el)

    azi = np.cumsum(np.minimum(abs(np.diff(az)), abs(np.diff((az - 180) % 360))))
    time = np.concatenate((np.zeros(1), azi / (rpm * 6)))
    
    data = np.ma.masked_array(out.astype(np.float64))
    for i in range(nfields):  
        data[...,i] = sf[i,0] * (np.ma.masked_equal(data[...,i],sf[i,3])-sf[i,1]) / sf[i,2]
    
    return np.arange(ngates) *dr , az, el, time, data, position


def scans_to_volume(files, position = None, azimuth = 0, verbose = False):
    stime = pd.to_datetime('_'.join((files[0].split('/')[-1]).split('_')[slice(1,3)]), format = '%Y%m%d_%H%M%S')
    field_names = ['RAIN', 'DBZH', 'VRADH', 'ZDR', 'KDP', 'PHIDP', 'RHOHV', 'WRADH', 'QC',] 
    els, azs, ds, times, ctime = None, None, None, None, 0
    sweep_end_ray_index = []
    for i, file in tqdm(enumerate(sorted(files))) if verbose else enumerate(sorted(files)):
        rs, az, el, time, data, position = scn_to_sweep(file, position = position)
        sweep_end_ray_index.append(az.shape[0])
        offset = np.amax(time) + 1 # arbitrary time for antenna realignment
        time = time + ctime
        ctime += offset
        azs, els, times = [v if vs is None else np.concatenate((vs, v)) for vs, v in zip([azs, els, times], [az, el, time])]
        ds = data if ds is None else np.concatenate((ds, data), axis=0)
            
    sweep_end_ray_index = np.cumsum(sweep_end_ray_index)
    sweep_start_ray_index = np.concatenate((np.zeros(1), sweep_end_ray_index[:-1])).astype(int)

    # check if in ppi mode
    fixed_angle = []
    for si, sf in zip(sweep_start_ray_index, sweep_end_ray_index):
        el = els[si:sf]
        assert len(np.unique(el)) ==1 
        fixed_angle.append(np.unique(el)[0])
    fixed_angle = np.array(fixed_angle)
    azs = (azs - azimuth) % 360
    
    fields = {}
    for i, name in enumerate(field_names):
        fields[name] = {'data': ds[...,i]}
        fields[name]['coordinates'] = 'elevation azimuth range'
        fields[name]['_FillValue'] = ds.fill_value
        
    # define stuff to initialize radar object
    radar_time = {'units': 'seconds since {}'.format(stime.strftime('%Y-%m-%dT%H:%M:%S%zZ')),
        'data' : times, '_FillValue': ds.fill_value}
    _range = {'units': 'meters', 'standard_name': 'projection_range_coordinate', 'long_name': 'range_to_measurement_volume', 
         'axis': 'radial_range_coordinate', 'spacing_is_constant': 'true', 'data': rs}
    metadata = {'Description': 'XPOL'}
    sweep_number = {'units': 'count', 'standard_name': 'sweep_number', 'long_name': 'Sweep number', 
                    'data': np.arange(len(fixed_angle))}
    sweep_mode = {'units': 'unitless', 'standard_name': 'sweep_mode', 'long_name': 'Sweep mode', 
                  'data': np.array(['azimuth_surveillance']*len(fixed_angle))}
    fixed_angle = {'long_name': 'Target angle for sweep', 'units': 'degrees', 'standard_name': 'target_fixed_angle', 
                   'data': fixed_angle}
    sweep_start_ray_index = {'long_name': 'Index of first ray in sweep, 0-based', 'units': 'count', 
                             'data': sweep_start_ray_index}
    sweep_end_ray_index = {'long_name': 'Index of last ray in sweep, 0-based', 'units': 'count', 
                            'data': sweep_end_ray_index-1}
    latitude = {'long_name': 'Latitude', 'standard_name': 'Latitude', 'units': 'degrees_north', 
                'data': np.array([position[0]])}
    longitude =  {'long_name': 'Longitude', 'standard_name': 'Longitude', 'units': 'degrees_east', 
                  'data': np.array([position[1]])}
    altitude = {'long_name': 'Altitude', 'standard_name': 'Altitude', 'units': 'meters', 'positive': 'up', 
                'data': np.array([0])}
    azimuth = {'units': 'degrees', 'standard_name': 'beam_azimuth_angle', 'long_name': 'azimuth_angle_from_true_north', 
               'axis': 'radial_azimuth_coordinate', 'comment': 'Azimuth of antenna relative to true north', 'data': azs.flatten()}
    elevation = {'units': 'degrees', 'standard_name': 'beam_elevation_angle', 'long_name': 'elevation_angle_from_horizontal_plane', 
                 'axis': 'radial_elevation_coordinate', 'comment': 'Elevation of antenna relative to the horizontal plane', 
                 'data': els.flatten()}
    radar = pyart.core.Radar(radar_time, _range, fields, metadata, 'ppi', latitude, longitude, altitude, 
                             sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index, sweep_end_ray_index, azimuth, elevation)
        
    return radar

def get_volumes_from_dir(dirr, start = None, end = None):
    scans = [file for file in sorted(os.listdir(dirr)) if file.split('.')[-1] == 'scn']
    times = pd.to_datetime(['_'.join(f.split('_')[slice(1,3)]) for f in scans], format = '%Y%m%d_%H%M%S')
    if start is not None:
        times = times[times >= start]
    if end is not None:
        times = times[times <= end]
    times = pd.to_datetime(np.unique(times)).strftime('%Y%m%d_%H%M%S')
    scans = [[dirr+'/'+file for file in scans if '_'.join(file.split('_')[slice(1,3)]) == time] for time in times]
    return scans