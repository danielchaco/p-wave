# # Functions to perform P-wave seismogram estimate of Vsz
# Based on P_wave_seismogram.py by Meibai Li, 04/14/2022

# Daniel M. Chacon
# daniel.chacon@upr.edu
# University of Puerto Rico - Mayagüez

# **Description:** This python script includes functions used to perform the P-wave seismogram method to estimate Vsz, which represents the averaged shear-wave velocity over depth z. The use of the functions is demonstrated in the Jupyter notebook `P-wave seismogram method_example.ipynb`.
# ** the Meibai Li script was adapted to read gse files then used them to process the records, extractplot original data,

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, tukey, find_peaks
from math import radians, sin, cos, atan2, sqrt, degrees, atan
import ctypes as C  # NOQA
import warnings
from obspy import UTCDateTime
from obspy.core.util.libnames import _load_cdll
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import BytesIO

# GSE files: parsing and decoding
# modification on libgse2.py to read all the channels
clibgse2 = _load_cdll("gse2")

clibgse2.decomp_6b_buffer.argtypes = [
    C.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags='C_CONTIGUOUS'),
    C.CFUNCTYPE(C.c_char_p, C.POINTER(C.c_char), C.c_void_p), C.c_void_p]
clibgse2.decomp_6b_buffer.restype = C.c_int

clibgse2.rem_2nd_diff.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_int]
clibgse2.rem_2nd_diff.restype = C.c_int

clibgse2.check_sum.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_int, C.c_int32]
clibgse2.check_sum.restype = C.c_int  # do not know why not C.c_int32

clibgse2.diff_2nd.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_int, C.c_int]
clibgse2.diff_2nd.restype = C.c_void_p

clibgse2.compress_6b_buffer.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_int,
    C.CFUNCTYPE(C.c_int, C.c_char)]
clibgse2.compress_6b_buffer.restype = C.c_int

def _str(s):
    return s.strip()

GSE2_FIELDS = [
    # local used date fields
    ('year', 5, 9, int),
    ('month', 10, 12, int),
    ('day', 13, 15, int),
    ('hour', 16, 18, int),
    ('minute', 19, 21, int),
    ('second', 22, 24, int),
    ('microsecond', 25, 28, int),
    # global ObsPy stats names
    ('station', 29, 34, _str),
    ('channel', 35, 38, lambda s: s.strip().upper()),
    ('gse2.auxid', 39, 43, _str),
    ('gse2.datatype', 44, 48, _str),
    ('npts', 48, 56, int),
    ('sampling_rate', 57, 68, float),
    ('calib', 69, 79, float),
    ('gse2.calper', 80, 87, float),
    ('gse2.instype', 88, 94, _str),
    ('gse2.hang', 95, 100, float),
    ('gse2.vang', 101, 105, float),
]

def parse_sta2(line):
    """
    Parses a string with a GSE2 STA2 header line.
    Official Definition::
        Position Name     Format    Description
           1-4   "STA2"   a4        Must be "STA2"
          6-14   Network  a9        Network identifier
         16-34   Lat      f9.5      Latitude (degrees, S is negative)
         36-45   Lon      f10.5     Longitude (degrees, W is negative)
         47-58   Coordsys a12       Reference coordinate system (e.g., WGS-84)
         60-64   Elev     f5.3      Elevation (km)
         66-70   Edepth   f5.3      Emplacement depth (km)
    Corrected Definition (end column of "Lat" field wrong)::
        Position Name     Format    Description
           1-4   "STA2"   a4        Must be "STA2"
          6-14   Network  a9        Network identifier
         16-24   Lat      f9.5      Latitude (degrees, S is negative)
         26-35   Lon      f10.5     Longitude (degrees, W is negative)
         37-48   Coordsys a12       Reference coordinate system (e.g., WGS-84)
         50-54   Elev     f5.3      Elevation (km)
         56-60   Edepth   f5.3      Emplacement depth (km)
    However, many files in practice do not adhere to these defined fixed
    positions. Here are some real-world examples:
    >>> l = "STA2           -999.0000 -999.00000              -.999 -.999"
    >>> for k, v in sorted(parse_sta2(l).items()):  \
            # doctest: +NORMALIZE_WHITESPACE
    ...     print(k, v)
    coordsys
    edepth -0.999
    elev -0.999
    lat -999.0
    lon -999.0
    network
    >>> l = "STA2 ABCD       12.34567   1.234567 WGS-84       -123.456 1.234"
    >>> for k, v in sorted(parse_sta2(l).items()):
    ...     print(k, v)
    coordsys WGS-84
    edepth 1.234
    elev -123.456
    lat 12.34567
    lon 1.234567
    network ABCD
    """
    header = {}
    try:
        lat = line[15:24].strip()
        if lat:
            lat = float(lat)
        else:
            lat = None
        lon = line[25:35].strip()
        if lon:
            lon = float(lon)
        else:
            lon = None
        elev_edepth = line[48:].strip().split()
        elev, edepth = elev_edepth or (None, None)
        if elev:
            elev = float(elev)
        else:
            elev = None
        if edepth:
            edepth = float(edepth)
        else:
            edepth = None
        header['network'] = line[5:14].strip()
        header['lat'] = lat
        header['lon'] = lon
        header['coordsys'] = line[36:48].strip()
        header['elev'] = elev
        header['edepth'] = edepth
    except Exception:
        msg = 'GSE2: Invalid STA2 header, ignoring.'
        warnings.warn(msg)
        return {}
    else:
        return header
    
def uncompress_cm6(f, n_samps):
    """
    Uncompress n_samps of CM6 compressed data from file pointer fp.
    :type f: file
    :param f: File Pointer
    :type n_samps: int
    :param n_samps: Number of samples
    """
    def read83(cbuf, vptr):  # @UnusedVariable
        line = f.readline()
        if line == b'':
            return None
        # avoid buffer overflow through clipping to 82
        sb = C.create_string_buffer(line[:82])
        # copy also null termination "\0", that is max 83 bytes
        C.memmove(C.addressof(cbuf.contents), C.addressof(sb), len(line) + 1)
        return C.addressof(sb)

    cread83 = C.CFUNCTYPE(C.c_char_p, C.POINTER(C.c_char), C.c_void_p)(read83)
    if n_samps == 0:
        data = np.empty(0, dtype=np.int32)
    else:
        # aborts with segmentation fault when n_samps == 0
        data = np.empty(n_samps, dtype=np.int32)
        n = clibgse2.decomp_6b_buffer(n_samps, data, cread83, None)
        if n != n_samps:
            raise GSEUtiError("Mismatching length in lib.decomp_6b")
        clibgse2.rem_2nd_diff(data, n_samps)
    return data

# based on ObsPy
def read_GSE2(fh):
    """
    Reads GSE2 headeres from file pointer and returns it as list of dictionaries.
    It also uncompress the data assuming a CM6 compression.
    The method searches for the next available WID2 field beginning from the
    current file position.
    """
    headers = []
    # search for WID2 field
    line = fh.readline()
    while line:
        if line.startswith(b'WID2'):
            # valid GSE2 header
            header = {}
            header['gse2'] = {}
            for key, start, stop, fct in GSE2_FIELDS:
                try:
                    value = fct(line[slice(start, stop)])
                except:
                    value = np.nan
                if 'gse2.' in key:
                    header['gse2'][key[5:]] = value
                else:
                    header[key] = value
            # convert and remove date entries from header dict
            header['microsecond'] *= 1000
            date = {k: header.pop(k) for k in
                    "year month day hour minute second microsecond".split()}
            header['starttime'] = UTCDateTime(**date)
            # search for STA2 line (mandatory but often omitted in practice)
            # according to manual this has to follow immediately after WID2
            pos = fh.tell()
            line = fh.readline()
            if line.startswith(b'STA2'):
                header2 = parse_sta2(line)
                header['network'] = header2.pop("network")
                header['gse2'].update(header2)
            # in case no STA2 line is encountered we need to rewind the file pointer,
            # otherwise we might miss the DAT2 line afterwards.
            else:
                fh.seek(pos)
            # Py3k: convert to unicode
            header['gse2'] = dict((k, v.decode()) if isinstance(v, bytes) else (k, v)
                                  for k, v in header['gse2'].items())
            header = dict((k, v.decode()) if isinstance(v, bytes) else (k, v)
                for k, v in header.items())
            
            ## get data assuming CM6 datatype            
            header['data'] = uncompress_cm6(fh, header['npts'])
            
            headers.append(header)
        line = fh.readline()
    return headers

def loadGSE(gse_path_url):
    '''
    returns a dataframe with gse data and metadata
    '''
    if 'https://' in gse_path_url:
        r = requests.get(gse_path_url, allow_redirects=True)
        fh = BytesIO(r.content)
        headers = read_GSE2(fh)
    else:
        with open(gse_path_url, 'rb') as fh:
            headers = read_GSE2(fh)
    lista = [{**dictionary['gse2'],**dictionary} for dictionary in headers]
    df = pd.DataFrame(lista)
    df.starttime = pd.to_datetime(df.starttime.astype('string'))
    df.drop(columns=['gse2'],inplace=True)
    return df

def plotWaves(df, get_fig = False):
    '''
    from df get data to plot the waves
    '''
    df = df.copy()
    df.reset_index(drop=True,inplace=True)
    fig = make_subplots(rows=len(df), cols=1)
    for i in df.index:
        data = df.data[i]
        time = df.starttime[i]+pd.to_timedelta([j/df.sampling_rate[i] for j in range(len(data))], unit='s')
        fig.add_trace(go.Scatter(x=time, y=data, name= df.station[i]+' '+df.channel[i]),row=i+1, col=1)
    fig.update_layout(height=150*len(df), width=1000, font_family = 'Times New Roman', margin=dict(l=0, r=0, t=0, b=0))
    fig.show()
    if get_fig:
        return fig

## some modifications from P_wave_seismogram.py 
# The following functions are used to solve for ray parameter p
def f(z, *data):
    '''
    # Purpose: Define a series of equations (fs) to be solved in solve_p
    # Inputs:
        z: a list of unknowns
        data: a tuple of input parameters that consist of:
          thickness of crustal layers
          p-wave velocity of crustal layers
          epicentral distance
          number of crustal model layers used to estimate p
    # Outouts:
        fs: equations used in solve_p
    '''
    # Initiate an empty list for equations
    fs = []
    # Epicentral distance
    R = data[-2]
    # Number of layers to be used
    N = data[-1]
    # Create a list of unknowns: horizontal distance traveled in each layer
    Rlist = []
    for i in range(N):
        Rlist.append(z[i])
    # Append fs with geometric relationship between every two layers
    # N-1 equations are written
    for c_id in range(N - 1):
        r = Rlist[c_id]
        r_next = Rlist[c_id + 1]
        d = data[c_id]
        d_next = data[c_id + 1]
        vp = data[N + c_id]
        vp_next = data[N + c_id + 1]
        f = sin(atan(r / d)) / vp - sin(atan(r_next / d_next)) / vp_next
        fs.append(f)
    # Add 1 equation: the sum of all r in Rlist is equal to epicentral distance
    fs.append(sum(Rlist) - R)
    return fs
 
def solve_p(D, R, thickness, Vps):
    '''
    # Purpose: solve for ray parameter p given a multi-layer crustal model
    # Inputs:
        D: focal depth of the earthquake (km)
        R: epicentral distance of the earthquake (km)
        thickness: a list of thickness of each layer in the crustal model
        Vps: a list of p-wave velocity of each layer in the crustal model
    # Outputs:
        p: estimated ray parameter
        Rlist: list of horizontal distances traveled in each layer by the path
        Dlist: list of thickness of layers traveled by the path
        Vplist: list of P-wave velocity of layers traveled by the path
    '''
    # Find the number of layers to be used for computing ray parameter p
    N = 0
    while D > sum(thickness[:N]):
        N += 1  

    # Create a list of thickness of the N layers used
    Dlist = thickness[0: N - 1]
    Dlist.append(D - sum(thickness[0: N - 1]))

    # Create a list of initial guess of distances
    iniguess = []
    for i in range(0, len(Dlist)-1):
        iniguess.append(0.0)
    iniguess.append(R)

    # Create a list of P-wave velocities of the N layers used
    Vplist = Vps[0: N]

    # Make a tuple of input parameters (used in function f)
    data = tuple(Dlist) + tuple(Vplist) + tuple([R]) + tuple([N])
    
    # Solve for the Rs
    Rlist = fsolve(f, iniguess, args = data)

    # Compute ray parameter p
    p = sin(atan(Rlist[0]/Dlist[0]))/Vplist[0]
    
    return p, Rlist, Dlist, Vplist


# In[3]:


# This equation is used to load txt file containing records
def loadtxt(filepath):
    '''
    # Purpose: load the records from txt file
    # Inputs:
        filepath: path of the txt file that is to be loaded
    # Outputs:
        data: ground motion record
        dt: sampling time interval
        datatype: 'accl', 'vel', or 'disp'
    # Notes:
        This function need to be modified based on the type of record.
    '''
    # Initiate a list to store lines of txt file
    lines = []
    with open(filepath) as fp:
        line = True
        while line:
            line = fp.readline().rstrip()
            lines.append(line)
            
    # get metadata of the record
    info = lines[0]
    dt = float(info.split()[3]) * 10 ** (-6)  # check the sampling interval (second)
    
    # Check unit of the time series (m/s, m/s^2, or m)
    unit = lines[0][lines[0].find('('):]
    if unit.count('SECOND') == 1:
        datatype = 'vel'
    elif unit.count('SECOND') == 2:
        datatype = 'accl' 
    else:
        datatype = 'disp'
        
    # Create a list of time series data
    data = []
    # Write content of lines into the list
    lines = lines[2:-1]
    if len(lines[-1].split()) == 2:   
        for i in range(0, len(lines)):
            record1 = float(lines[i].split()[0])
            record2 = float(lines[i].split()[1])
            data.extend([record1, record2])
    elif len(lines[-1].split()) == 1:
        for i in range(0, len(lines) - 1):
            record1 = float(lines[i].split()[0])
            record2 = float(lines[i].split()[1])
            data.extend([record1, record2])
        data.extend([float(lines[-1])])
    data = np.array(data)
        
    return data, dt, datatype


# In[74]:


# The following functions are used to process records
def F_diff(data, dt):
    # Differentiation
    data_diff = np.diff(data) / dt
    return data_diff

def F_int(data, dt):
    # Integration
    data_int = cumulative_trapezoid(data, initial=0) * dt
    return data_int
    
def F_filter(order, lowPassF, highPassF, acc, dt):
    # Bandpass Butterworth filtering
    b, a = butter(order, [highPassF * (2 * dt), lowPassF * (2 * dt)], 'band')
    acc = filtfilt(b, a, acc)
    return acc

def dataprocess(data, dt, datatype):
    '''
    # Purpose: This function is used for batch processing acceleration/velocity/displacement time series.
    # Inputs: 
        data: signal to be processed
        dt: sampling interval
        datatype: 'accl', 'vel', or 'disp'
    # Outputs:
        vel_processed: processed velocity time series
        t: corresponding time sequence.
    # Notes:
        This function does not include instrument response removal.
        The Prism Guide is used as reference of the processing procedure.
    '''
    # Get sampling frequency
    freq = 1 / dt
    # Data processing procedures are different for seismometer data (velocity) and accelerometer data (acceleration)
    if datatype == 'disp':
        accl = F_diff(F_diff(data, dt), dt)
        vel_processed, t = dataprocess(accl, dt, 'accl')
    elif datatype == 'vel':
        accl = F_diff(data, dt)
        vel_processed, t = dataprocess(accl, dt, 'accl')       
    elif datatype == 'accl':
        accl = data
        numaccl = len(accl)
        t = np.array(range(0, numaccl)) * dt

        # Remove mean from acceleration
        accl = accl - np.mean(accl)

        # Trapezoidal rule is used to integrated from acceleration to velocity and from velocity to displacement
        vel = F_int(accl, dt)
        disp = F_int(vel, dt)

        # Linear line is used to fit the trend in velocity:
        term = 1.0
        velCoeffs = np.polyfit(t, vel, term)

        # Remove the derivative of the best fit trend in velocity from the acceleration time series
        correction = 0.
        for j in range(0, len(velCoeffs) - 1):
            correction = correction + (velCoeffs[j] * (term - j)) * pow(t, term - 1 -j)
        accl = accl - correction

        # Acceleration record is then integrated to velocity
        vel = F_int(accl, dt)
        disp = F_int(vel, dt)

        # Quality check for velocity (Here I just check the initial value of velocity)
        # Make correction to initial value of velocity by checking if there's a linear trend in displacement
        term = 1.0
        dispCoeffs = np.polyfit(t, disp, term)
        vel = vel - dispCoeffs[0]

        # Integrate velocity again to get displacement
        disp = F_int(vel, dt)

        # Apply cosine tapering at the beginning and end of the record, then pad zeros at the two ends of the record
        # The cosine taper length is set to be 0.1 of the entire record length
        # Create and apply tukey window
        win = tukey(len(accl), 0.1)
        accl = accl * win

        # Typical number of zeros to be padded
        order = 4.0
        f1c = 0.1  # Low-cut corner frequency
        Tpad = 1.5 * order / f1c * freq
        zeropad = np.zeros(round(Tpad / 2))
        acclpad = np.append(zeropad, accl)
        acclpad = np.append(acclpad, zeropad)

        # Find new length of acceleration time series and create new time array
        numaccl_pad = len(acclpad)
        time_pre = np.array(range(0, len(zeropad))) * (-1) * dt
        time_post = (np.array(range(0, len(zeropad))) + numaccl) * dt
        timepad = np.append(time_pre, t)
        timepad = np.append(timepad, time_post)

        # Bandpass filtering

        order = 4.0
        lowPassF = min(25.0, 0.4 * freq)
        highPassF = 0.3
        acclfilt = F_filter(order, lowPassF, highPassF, acclpad, dt)

        # Computation of Velocity and Displacement
        velpad = F_int(acclfilt, dt)
        disppad = F_int(velpad, dt)

        # Chop the time series to their original length
        accl_processed = acclfilt[len(zeropad):(len(acclfilt) - len(zeropad))]
        vel_processed = velpad[len(zeropad):(len(acclfilt) - len(zeropad))]
        disp_processed = disppad[len(zeropad):(len(acclfilt) - len(zeropad))]

        vel_processed = vel_processed - np.mean(vel_processed)
        t = np.array(range(0, len(vel_processed))) * dt
    
    return vel_processed, t


# In[84]:


# Rotate horizontal components of velocity time series to radial direction
def rotate(data1, data2, azi1, azi2, Evtlat, Evtlon, Stalat, Stalon):
    '''
    # Purpose: Rotate the horizontal components and obtain epicentral distance and azimuth
    # Input:
        data1: First horizontal component
        data2: Second horizontal component
        azi1: Azimuth angle of data1 (direction from north)
        azi2: Azimuth angle of data2 (direction from north)
        Evtlat, Evtlon: latitude, longitude of the event (epicenter) in degrees
        Stalat, Stalon: latitude, longitude of the station in degrees
    # Output:
        datar: radial component
        distance: epicentral distance (km)
        azimuth: azimuth angle of station relative to event in degrees
    # Notes:
        Haversine formula is used to compute distance and azimuth.
        If north and east components are used, azi1 and azi2 are 0 and 90.
    '''
    R = 6373.0    # approximate radius of earth in km

    Evtlat = radians(Evtlat)
    Evtlon = radians(Evtlon)
    Stalat = radians(Stalat)
    Stalon = radians(Stalon)

    dlon = Stalon - Evtlon
    dlat = Stalat - Evtlat  

    a = sin(dlat / 2)**2 + cos(Evtlat) * cos(Stalat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance = R * c
    
    azimuth = degrees(atan2((sin(dlon) * cos(Stalat)), (cos(Evtlat) * sin(Stalat) - sin(Evtlat) * cos(Stalat) * cos(dlon))))
    
    datar = (data1 * cos(radians(azi1 - azimuth)) + data2 * cos(radians(azi2 - azimuth)))
    return datar, distance, azimuth


# In[86]:


def process_records(filepath1, filepath2, filepath3, azi1, azi2, Evtlat, Evtlon, Stalat, Stalon):
    '''
    # Purpose: This function read data, check the SNR, and rotate the data
    # Inputs:
        filepath1, filepath2: Path of two horizontal components of velocity time series
        filepath3: Path of vertical component of velocity time series
        azi1, azi2: Azimuth angles of data1 and data2
        Evtlat, Evtlon: latitude, longitude of the event (epicenter) in degrees
        Stalat, Stalon: latitude, longitude of the station in degrees
    # Outputs:
        t: time (s)
        dt: sampling rate (s)
        ur, uz: radial and vertical components of data
        Epidist: epicentral distance (km)
        SNRpass: whether the SNR of the data is satisfactory
        start: appropximate start time of the earthquake event (s)      
    '''
    data1, dt, datatype = loadtxt(filepath1)
    data2, _, _ = loadtxt(filepath2)
    data3, _, _ = loadtxt(filepath3)

    # Check lengths of records:
    if len(data1) == len(data2) and len(data1) == len(data3):
        pass
        # print('All records are of the same length')
    else:
        print('Error: The length of the records are different')
  
    # Process all three components (Need to check if the data is in velocity or acceleration if batch processing is needed)    
    u1, t = dataprocess(data1, dt, datatype)
    u2, _ = dataprocess(data2, dt, datatype)
    uz, _ = dataprocess(data3, dt, datatype)

    # Rotate two horizontal components to radial direction
    ur, Epidist, _ = rotate(u1, u2, azi1, azi2, Evtlat, Evtlon, Stalat, Stalon)
    
    SNRpass = 'y'
    start = 120
    '''
    # Uncomment if need to check signal to noise ratio
    
    # Use uz to compute signal to noise ratio:
    freq = 1 / dt
    t_signal = [120, 160]   # Start and end time of signal used for computing SNR
    t_noise = [95, 115]     # Start and end time of noise used for computing SNR
    # Slice signal and noise used for computing SNR
    signal = uz[int(freq * t_signal[0]):int(freq * t_signal[1])]
    noise = uz[int(freq * t_noise[0]):int(freq * t_noise[1])]
    snr = np.std(signal)**2 / np.std(noise)**2
    if snr < 10:
        # The pre-event noise is too large for the record to be considered
        SNRpass = 'n'
    else:
        # Manual checking of SNR
        plt.figure(figsize = (20, 5))
        plt.plot(t, uz, label = 'Vertical')
        plt.xlabel('time (sec)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(which = 'both')
        plt.show()
        start = float(input('Please input the approximate start time of the earthquake (s): '))

        # Zoom in to the begining of the event
        plt.figure(figsize = (20, 5))
        index_start = int((start - 10.0) * freq)
        index_end = int((start + 10.0) * freq)
        plt.plot(t[index_start:index_end], uz[index_start:index_end], label = 'Vertical zoomed in')
        plt.xlabel('time (sec)')
        plt.ylabel('Velocity (m/s)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
        SNRpass = input('Does the record meet the SNR requirement (y or n): ')  # SNRpass = 'n' for records that are too noisy
    '''
    return t, dt, ur, uz, Epidist, SNRpass, start


def process_records2(df, Evtlat, Evtlon, datatype = 'vel'):
    '''
    Similar to process_records just changes the input.
    # Purpose: This function read data, check the SNR, and rotate the data
    # Inputs:
        df: pandas dataframe with horizontal and vertical components of velocity time 
        series. It contains as well the metadata of the station: Stalat, Stalon.
        azi1, azi2: Azimuth angles of first and second horizontal channel resp.
        Evtlat, Evtlon: latitude, longitude of the event (epicenter) in degrees
    # Outputs:
        t: time (s)
        dt: sampling rate (s)
        ur, uz: radial and vertical components of data
        Epidist: epicentral distance (km)
        SNRpass: whether the SNR of the data is satisfactory
        start: appropximate start time of the earthquake event (s)      
    '''
    df.reset_index(inplace=True,drop=True)
    data1 = df.data[1]
    azi1 = df.hang[1]
    data2 = df.data[2]
    azi2 = df.hang[2]
    data3 = df.data[0]
    dt = 1/df.sampling_rate[0]
    Stalat = df.lat[0]
    Stalon = df.lon[0]

    # Check lengths of records:
    if len(data1) == len(data2) and len(data1) == len(data3):
        pass
        # print('All records are of the same length')
    else:
        print('Error: The length of the records are different')
  
    # Process all three components (Need to check if the data is in velocity or acceleration if batch processing is needed)    
    u1, t = dataprocess(data1, dt, datatype)
    u2, _ = dataprocess(data2, dt, datatype)
    uz, _ = dataprocess(data3, dt, datatype)

    # Rotate two horizontal components to radial direction
    ur, Epidist, _ = rotate(u1, u2, azi1, azi2, Evtlat, Evtlon, Stalat, Stalon)
    
    SNRpass = 'y'
    start = 120
    '''
    # Uncomment if need to check signal to noise ratio
    
    # Use uz to compute signal to noise ratio:
    freq = 1 / dt
    t_signal = [120, 160]   # Start and end time of signal used for computing SNR
    t_noise = [95, 115]     # Start and end time of noise used for computing SNR
    # Slice signal and noise used for computing SNR
    signal = uz[int(freq * t_signal[0]):int(freq * t_signal[1])]
    noise = uz[int(freq * t_noise[0]):int(freq * t_noise[1])]
    snr = np.std(signal)**2 / np.std(noise)**2
    if snr < 10:
        # The pre-event noise is too large for the record to be considered
        SNRpass = 'n'
    else:
        # Manual checking of SNR
        plt.figure(figsize = (20, 5))
        plt.plot(t, uz, label = 'Vertical')
        plt.xlabel('time (sec)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(which = 'both')
        plt.show()
        start = float(input('Please input the approximate start time of the earthquake (s): '))

        # Zoom in to the begining of the event
        plt.figure(figsize = (20, 5))
        index_start = int((start - 10.0) * freq)
        index_end = int((start + 10.0) * freq)
        plt.plot(t[index_start:index_end], uz[index_start:index_end], label = 'Vertical zoomed in')
        plt.xlabel('time (sec)')
        plt.ylabel('Velocity (m/s)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
        SNRpass = input('Does the record meet the SNR requirement (y or n): ')  # SNRpass = 'n' for records that are too noisy
    '''
    return t, dt, ur, uz, Epidist, SNRpass, start

# In[87]:


def comp_ratio(t, ur, uz, freq, start=120):
    '''
    # Purpost: This function is used to pick Ur and Uz
    # Inputs:
        t: time (s)
        ur, uz: radial and vertical components of data
        freq: sampling frequency
        start: appropximate start time of the earthquake event (s), use 120s as default
    # Outputs:
        ratio: Ratio between radial to vertical components at the first peak in uz
        arri: index of first peak
        index_start, index_end: start and end index of data used to find Ur/Uz value (may be used for plotting purpose)
    '''
    # Plot the full time domain
    plt.figure(figsize = (20, 5))
    plt.plot(t, ur, label = 'Radial')
    plt.plot(t, uz, label = 'Vertical')
    plt.xlabel('time (sec)')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    # Manually pick the first peak in vertical component
    pick_ok = 'n'
    while pick_ok != 'y':
        # Zoom in to the approximate time of P-wave arrival
        zoom_to = start
        plt.figure(figsize = (20, 5))
        index_start = int((zoom_to - 10.0) * freq)
        index_end = int((zoom_to + 10.0) * freq)
        plt.plot(t[index_start:index_end], uz[index_start:index_end], label = 'Vertical')
        plt.plot(t[index_start:index_end], ur[index_start:index_end], label = 'Radial')
        plt.xlabel('time (sec)')
        plt.ylabel('Velocity (m/s)')
        plt.legend(loc='upper right')
        plt.grid(True)  
        plt.show()
        # Zoom in further:
        zoom_to = float(input('Please input the time of P-wave arrival (about 0.5 sec precision):'))
        plt.figure(figsize = (20, 5))
        index_start = int((zoom_to - 2.5) * freq)
        index_end = int((zoom_to + 2.5) * freq)
        plt.plot(t[index_start:index_end], uz[index_start:index_end], label = 'Vertical')
        plt.plot(t[index_start:index_end], ur[index_start:index_end], label = 'Radial')
        plt.xlabel('time (sec)')
        plt.ylabel('Velocity (m/s)')
        plt.legend(loc='upper right')
        plt.grid(True) 
        plt.show()

        # Choose the begin time to find peaks:
        begin_time = float(input('Please input the time begin from when you want to find peaks:'))
        height_limit = float(input('Please input the amplitude threshold to initiate the peak search:'))

        # Find the first peak after P-wave arrival from vertical component:
        begin_index = int(begin_time * freq)
        end_index = index_end
        peaks_positive, _ = find_peaks(uz[begin_index:end_index], height = height_limit, threshold = None, distance=5)
        peaks_negative, _ = find_peaks(-uz[begin_index:end_index], height = height_limit, threshold = None, distance=5)

        if len(peaks_positive) == 0:
            arri = peaks_negative[0] + begin_index
        elif len(peaks_negative) == 0:
            arri = peaks_positive[0] + begin_index
        else:
            arri = min(peaks_positive[0], peaks_negative[0]) + begin_index
        Uz = uz[arri]
        Ur = ur[arri] 

        # Indicate the arrivals on the time series
        plt.figure(figsize = (20, 5))
        plt.plot(t[index_start:index_end], uz[index_start:index_end], label = 'Vertical')
        plt.plot(t[index_start:index_end], ur[index_start:index_end], label = 'Radial')
        plt.scatter([t[arri], t[arri]], [Ur, Uz], label = 'First peak of vertical velocity component', c = 'red')
        plt.xlabel('time (sec)')
        plt.ylabel('Velocity (m/s)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()    

        # Ask if the pick is satisfactory
        pick_ok = input('Does the pick look okay (y or n)? ')

    # Compute the ratio between the radial and vertical components
    ratio = abs(Ur / Uz)
    
    return ratio, arri, index_start, index_end


# In[89]:


# The following function is used to compute Vsz
def solve_Vsz(ratio, p):
    js = atan(ratio) / 2
    Vs = sin(js) / p
    return Vs, js


# In[ ]:


# Other functions:

def counts2vel(df, scale):
    '''
    converts the gse dataframe counts data to velocity according to the scale.
    if scale is a list, it is assumed that is sorted in the same way as the df.
    if not, all counts will be divided by the scale number.
    '''
    df = df.copy()
    df.reset_index(inplace=True,drop=True)
    data = []
    if type(scale) == list:
        if len(df) == len(scale):
            for j,i in enumerate(df.index):
                data.append(df.data[i] / scale[j])
        else:
            raise Exception("Sorry, lengths between scales and dataframe do not match.")
    elif type(scale) == dict:
        sg = scale['Sensitivity Gain']
        for i in df.index:
            data.append(df.data[i] / sg[df.channel[i]])
    else:
        for i in df.index:
            data.append(df.data[i] / scale)
    df.data = data
    return df

def prsnExtract_url(df,i,length=6):
    date = df['Date(UTC)'][i]
    time = df['Time'][i]
    url = f'https://worm.uprm.edu/cgi-bin/prsnExtract.cgi?network=All&starttime={date}\\T{time}&length={length}&format=gse'
    print(url)
    
    
    
def get_epicentral_distances(sol_paths, eventID, header = 'STA NET'):
    '''
    eventID: given by cat.ID[event_index]
    returns a dictionary with station/channel distance info
    '''
    dist_dict = {}
    for path in sol_paths:
        if str(eventID) in path:
            print(f'ID:\t{eventID}\npath:\t{path}')
            found = False
            with open(path,'r') as f:
                ls = f.readlines()
            for l in ls:
                if found:
                    try:
                        sta = l[sta_i:sta_i+5].strip()
                        if sta not in dist_dict.keys():
                            dist_dict[sta] = {}
                        chn = l[chn_i:chn_i+4].strip()
                        dist = float(l[dist_i:dist_i+5].strip())
                        dist_dict[sta][chn] = dist
                    except:
                        pass
                if header in l:
                    found = True
                    sta_i = l.find('STA')
                    chn_i = l.find('COM')
                    dist_i = l.find('DIST')
            return dist_dict
    print('There is no solution path for this event. Perfectly spherical Earth method can be used instead.')