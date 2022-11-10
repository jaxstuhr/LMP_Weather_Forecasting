import pickle, rasterio, pytz, s3fs, zarr, requests, tempfile, zipfile, glob, os, sys, xarray, bz2

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta
from solarpy import irradiance_on_plane

from scipy.interpolate import griddata

# Define temporal directory to store downloding chunks
path_to_temp = r"/Users/Guille/Desktop/caiso_power/data/temp/"

# Define output paths
path_to_oac = r"/Users/Guille/Desktop/caiso_power/output/actuals/"
path_to_ofc = r"/Users/Guille/Desktop/caiso_power/output/forecasts/"
path_to_oax = r"/Users/Guille/Desktop/caiso_power/output/auxiliary/"

# California region
lon_min = -125
lon_max = -112
lat_min = 32
lat_max = 43

# Grid resolution in degrees
delta = .125

repo      = sys.argv[1]
forecasts = int(sys.argv[2])
year      = int(sys.argv[3])
month     = int(sys.argv[4])
day       = int(sys.argv[5])
if forecasts == 1: cycle = int(sys.argv[6])

# Define url to online repository (i.e., bucket)
if repo == "Google":
    bucket = r"https://storage.googleapis.com/high-resolution-rapid-refresh"
    product = "wrfnat"
if repo == "Amazon":
    bucket  = r"https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    product = "wrfnat"
if repo == "Windows":
    bucket  = r"https://noaahrrr.blob.core.windows.net/hrrr"
    product = "wrfsfc"
# Constants for creating the full URL
sector  = "conus"
# Let's grab those variables:
keys_ = [":PRES:surface",
         ":PRATE:surface",
         ":DSWRF:surface",
         ":DLWRF:surface",
         ":DPT:2 m above ground",
         ":RH:2 m above ground",
         ":TMP:2 m above ground",
         ":UGRD:10 m above ground",
         ":VGRD:10 m above ground",
         ":UGRD:80 m above ground",
         ":VGRD:80 m above ground"]

# Derive Confort Metrics used in the literature
def _confort_metrics(T_, RH_, M_):

    # Transform units
    T_ -= 273.15
    M_ /= 1000.
    # Disfonfor Index - T [C] - RH [%]
    DI_ = (1.8 * T_ + 32.) + ( (.55 - .0055 * RH_) * (1.8 * T_ - 26.) )
    # Wind chill - Wind speed [km/h] - T [C]
    WC_ = 13.12 + (.06215 * T_) - (11.37 * (M_**.16)) + (.3965 * T_ * (M_**.16))
    # Cooling Degree HourHours - T [F]
    HCDH_ = T_ - 18.33

    return DI_ + 273.15, WC_ + 273.15, HCDH_

# Wind speed extrapolation at multiple altitures (10, 60, 80, 100, and 120 m)
def _extrapolate_wind(u_10_, v_10_, u_80_, v_80_):
    # Compute wind magnitude from both components
    def __mag(u_, v_):
        return np.sqrt(u_**2 + v_**2)
    # Compute power law
    def __power_law(m_1_, h_1, h_2, alpha):
        return m_1_ * (h_2/h_1)**2

    # Get wind speed
    m_10_ = __mag(u_10_, v_10_)
    m_80_ = __mag(u_80_, v_80_)
    # Compute power law exponent
    alpha = (np.log(m_10_) - np.log(m_80_))/(np.log(80.) - np.log(10.))
    # Compute wind speed applying power law
    m_60_  = __power_law(m_10_, 10., 60., alpha)
    m_100_ = __power_law(m_80_, 80., 100., alpha)
    m_120_ = __power_law(m_80_, 80., 120., alpha)

    return m_10_, m_60_, m_80_, m_100_, m_120_

def _ideal_solar_radiation(h_, lat_, year, month, day, hour):
    vnorm = np.array([0, 0, -1])  # plane pointing zenith
    #h = 0  # sea-level
    date = datetime(year, month, day, hour, 1)  # year, month, day, hour, minute
    #lat = -23.5  # southern hemisphere
    I_ = np.zeros((lat_.shape[0],))
    for i in range(lat_.shape[0]):
        if h_[i] < 0.:
            h_[i] = 0.
        I_[i] = irradiance_on_plane(vnorm, float(h_[i]), date, lat_[i])
    return I_

# Zulu Time - Which time frame should I use? I think 0 -> to 16:00 in Pacific Time give 3 hours (time buffer)
# of computing time to run the pipeline and make a prediction
# if online model and offline model is implemented 6 -> 20:00 would give a better approximation of the forecasting accuracy
def _NOAA_API(keys_, date, cycle, forecast, bucket, sector, product, W_ref_, index_):

    # Put it all together
    url = f"{bucket}/hrrr.{date}/{sector}/hrrr.t{cycle:02}z.{product}f{forecast:02}.grib2"
    print(url)
    # You can see it has a 1-indexed base line number, staring byte position, date, variable, atmosphere level,
    # and forecast description. The lines are colon-delimited.
    _req = requests.get(f"{url}.idx")
    idx_ = _req.text.splitlines()
    #print(*idx_, sep = "\n")

    X_ = []
    for key in keys_:
        # Find Variable
        variable_idx = [l for l in idx_ if key in l][0].split(":")
        # The line number values are 1-indexed, so we don't need to increment it to get the next list index,
        # but check we're not already reading the last line
        range_start = variable_idx[1]
        line_num    = int(variable_idx[0])
        next_line   = idx_[line_num].split(':') if line_num < len(idx_) else None
        range_end   = next_line[1] if next_line else None
        # Download Variable in a Temp. file
        _file = tempfile.NamedTemporaryFile(prefix = path_to_temp + "temp_", delete = False)
        _req  = requests.get(url, headers = {"Range": f"bytes={range_start}-{range_end}"}, stream = True)
        with _file as f: f.write(_req.content)
        #print(_file.name)

        # Load Variable and Stack in data Tensor
        _grib    = xarray.open_dataset(_file.name, engine = 'cfgrib', backend_kwargs = {'indexpath': ''})
        Z_       = _grib[list(_grib.keys())[0]].to_numpy()
        z_prime_ = Z_.flatten()[index_]
        # Perfom Interpolation
        z_ref_ = griddata(W_prime_prime_, z_prime_, (W_ref_[:, 0], W_ref_[:, 1]), method = 'nearest')

        X_ += [z_ref_[np.newaxis, ...]]
        os.remove(_file.name)

    return np.concatenate(X_, axis = 0)

market = ["ACTUAL", "DAM"][forecasts]
path   = [path_to_oac, path_to_ofc][forecasts]
# Get NOAA Operational forecast Database
database_ = zarr.open(s3fs.S3Map("s3://hrrrzarr", s3 = s3fs.S3FileSystem(anon = True)))

# Get NOAA grid Coordiantes
lat_ = database_["grid/HRRR_chunk_index.zarr"]["latitude"][...]
lon_ = database_["grid/HRRR_chunk_index.zarr"]["longitude"][...]

# Define the desired grid coordinates
W_prime_ = np.concatenate((lon_.flatten()[:, np.newaxis],
                           lat_.flatten()[:, np.newaxis]), axis = 1)

# Find coodiantes withint California region
idx_x_ = (W_prime_[:, 0] > lon_min) & (W_prime_[:, 0] < lon_max)
idx_y_ = (W_prime_[:, 1] > lat_min) & (W_prime_[:, 1] < lat_max)
index_ = idx_x_ & idx_y_

# Select California Region
W_prime_prime_ = W_prime_[index_, :]

# load Reference grid name
file_name = path_to_oax + 'ref_grid_{}_({},{})_({},{}).pkl'.format(delta, lon_min, lon_max, lat_min, lat_max)
with open(file_name, 'rb') as _f:
    W_ref_ = pickle.load(_f)
print(file_name)

# Compose saving name
file_name = path_to_oax + 'altitude_grid_{}_({},{})_({},{}).pkl'.format(delta, lon_min, lon_max, lat_min, lat_max)
with open(file_name, 'rb') as _f:
    h_ref_ = pickle.load(_f)
print(file_name)

# Set configuration to retrive forecast or actuals
if forecasts == 0:
    x_idx_     = [0, 2, 3, 4, 5, 6]
    cycles_    = np.arange(0, 24, 1, dtype = int)
    forecasts_ = [0]
else:
    x_idx_     = [0, 1, 2, 3, 4, 5, 6]
    cycles_    = [cycle]
    forecasts_ = np.arange(1, 37, 1, dtype = int)

# Retrive a year long data
_start_date = datetime(year, month, day)
_end_date   = datetime(year + 1, 1, 1)
N_days      = (_end_date - _start_date).days
print(N_days)

# Defining the timezone
_utc = pytz.timezone('UTC')
_ptz = pytz.timezone('America/Los_Angeles')

# Get daylong data request
_delta = timedelta(days = 1/24)

# Define date in is corresponding timezone
_utc_date = _utc.localize(_start_date)
#print(_utc_date)

# Loop over number of days
for i in range(N_days):

    # Get date in API request format
    time = f"{_utc_date.year:02}{_utc_date.month:02}{_utc_date.day:02}"

    if forecasts == 0: pickle_file_name = f"Weather_{time}.pkl"
    else:              pickle_file_name = f"Weather_{time}-{cycle:02}.pkl"

    if not os.path.isfile(path + pickle_file_name):
        print(time, market)

        # Check time difference between time zones
        _ptz_date = _utc_date.astimezone(_ptz)

        try:
            data_p_ = []
            # Loop over hours
            for cycle in cycles_:

                data_ = []
                # Loop over forecasting horizons
                for forecast in forecasts_:

                    print(_ptz_date)
                    # Add to list downloaded weather features for given forecasting interval
                    X_ = _NOAA_API(keys_, time, cycle, forecast, bucket, sector, product, W_ref_, index_)

                    # Get only useful convariates
                    X_p_ = X_[x_idx_, ...]

                    # Get extrapolated wind speed
                    M_10_, M_60_, M_80_, M_100_, M_120_ = _extrapolate_wind(X_[7, :], X_[8, :], X_[9, :], X_[10, :])

                    # Get wind speedx in matrix form
                    M_ = np.concatenate((M_10_[np.newaxis, ...],
                                         M_60_[np.newaxis, ...],
                                         M_80_[np.newaxis, ...],
                                         M_100_[np.newaxis, ...],
                                         M_120_[np.newaxis, ...]), axis = 0)

                    # Get confort metrics
                    DI_, WC_, HCDH_ = _confort_metrics(X_[6, :], X_[5, :], M_10_)

                    # Get confort metrics in matrix form
                    C_ = np.concatenate((DI_[np.newaxis, ...],
                                         WC_[np.newaxis, ...],
                                         HCDH_[np.newaxis, ...]), axis = 0)

                    # Get ideal solar radiation
                    GSI_ = _ideal_solar_radiation(h_ref_, W_ref_[:, 1], _ptz_date.year, _ptz_date.month, _ptz_date.day, _ptz_date.hour)

                    # Save weather features for the h-th hour
                    data_ += [np.concatenate((X_p_, M_, C_, GSI_[np.newaxis, ...]), axis = 0)[np.newaxis, ...]]

                    # Got to next day
                    _ptz_date += _delta

                # Save data with each weather feature for each grid point and each forecasting horizon
                data_p_ += [np.concatenate(data_, axis = 0)[np.newaxis, ...]]

            # Save data with each weather feature for each grid point, each forecasting horizon, and cycle
            data_pp_ = np.concatenate(data_p_, axis = 0)

            with open(path + pickle_file_name, 'wb') as _f:
                pickle.dump(data_pp_, _f, protocol = pickle.HIGHEST_PROTOCOL)
        except:
            print('Error: ', time, market)

    else:
        print('Exits: ', time, market)

    # Got to next day
    _utc_date += timedelta(days = 1)
