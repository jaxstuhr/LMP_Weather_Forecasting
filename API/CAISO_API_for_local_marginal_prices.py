import pickle, requests, zipfile, glob, os, sys

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, timedelta

# Path to folder where ca_node_locations.csv file is
path_to_oax = r"/Users/Guille/Desktop/caiso_power/output/auxiliary/"
# Path to folder where file processed .pkl files are stored
path_to_lmp = r"/Users/Guille/Desktop/caiso_power/output/LMPs/"

# California region to rank nodes with respect to
lat = 34.41
lon = -119.85
# Specify LMPs market
market = 'RTM'
# Time intervals in downloading batches
dt = 29

# Request specific parameters
year   = int(sys.argv[1])
month  = int(sys.argv[2])
i_node = int(sys.argv[3])

# GMT: Greenwich Mean Time
def _download_csv(url, zip_file_name = 'temp.zip'):
    print(url)
    # Request file on the url
    response = requests.get(url, stream = True)
    # Downalaod zip file from url
    with open(zip_file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size = 512):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    # unzip file
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall('')
    # Find unzip .csv in temp folder
    csv_file_name = glob.glob('*.csv')[0]
    # Open /csv file
    df_ = pd.read_csv(csv_file_name)
    # Remove temp files
    os.remove(csv_file_name)
    os.remove(zip_file_name)
    return df_

# Process LMPs request to the needed form
def _processing_data(INTVL_, data_):
    # Get only LMPs
    index_ = INTVL_['LMP_TYPE'] == 'LMP'
    # Get information associated with LMPs
    dates_     = INTVL_['OPR_DT'][index_].to_numpy()
    intervals_ = INTVL_['OPR_INTERVAL'][index_].to_numpy()
    values_    = INTVL_['VALUE'][index_].to_numpy()
    hours_     = INTVL_['OPR_HR'][index_].to_numpy()
    #print(intervals_.shape, values_.shape)
    # Loop over dates in the request
    for date in np.sort(np.unique(dates_)):
        # Select samples from a unique date
        idx_1_ = dates_ == date
        LMPs_ = []
        # Process LMPs by hour
        for hour in np.sort(np.unique(hours_[idx_1_])):
            # Get all data from that hour
            idx_2_ = hours_[idx_1_] == hour
            # Sort date by interval
            idx_3_ = np.argsort(intervals_[idx_1_][idx_2_])
            LMPs_.append(values_[idx_1_][idx_2_][idx_3_][:, np.newaxis])

        # Save only if all LMPs are available
        if len(LMPs_) == 24:
            data_[date] = np.concatenate(LMPs_, axis = 1)

    return data_

# Save CAISO data in a pickle file
def _save_CAISO_data_in_pickle_file(data_, file_name):
    # Compose saving name
    with open(file_name, 'wb') as _f:
        pickle.dump(data_, _f, protocol = pickle.HIGHEST_PROTOCOL)

bucket = r'http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6'

data_    = pd.read_csv(path_to_oax + r'ca_node_locations.csv')
nodes_   = data_['node_id']
lat_     = data_['lat']
lon_     = data_['long']
regions_ = data_['region']

# Select ranked node by distnace
idx    = np.argsort((lat_ - lat)**2 + (lon_ - lon)**2)[i_node]
node   = nodes_[idx]
lat    = lat_[idx]
lon    = lon_[idx]
region = regions_[idx]
print(node, region, lat, lon)

# Define url request for real-time 5-minutes LMPs
query  = 'PRC_INTVL_LMP'
query  = r'&queryname={}&version=3'.format(query)
market = '&market_run_id={}&node={}'.format(market, node)

file_name = path_to_lmp + region + r'_' + node + r'.pkl'
print(file_name)

day = 1

# Check if node-data exits for that date
if not os.path.isfile(file_name):
    # Start desired node dictionary
    data_         = {}
    data_['info'] = [nodes_[idx], lat_[idx], lon_[idx]]
else:
    data_ = pd.read_pickle(file_name)
    date  = list(data_.keys())[-1]
    year  = int(date[:4])
    month = int(date[5:7])
    day   = int(date[-2:])
    print('Exits until: ', year, month, day)

_start_date = datetime(year, month, day)

N_requests = int((datetime.now() - _start_date).days/dt)
print('No. requests: ', N_requests)

for i in range(N_requests):
    print('Pulling request No.', i + 1)

    try:
        _end_date  = _start_date + timedelta(dt)
        # Define date interval for url requester
        start_date = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}T00"
        end_date   = f"{_end_date.year:02}{_end_date.month:02}{_end_date.day:02}T23"
        period     = r'&startdatetime={}:00-0000&enddatetime={}:00-0000&'.format(start_date, end_date)
        # Download real-time 5-minutes LMPs
        url    = bucket + query + period + market
        INTVL_ = _download_csv(url, zip_file_name = 'temp.zip')
        # Processing requested data
        data_ = _processing_data(INTVL_, data_)
        # Save requested processed data
        _save_CAISO_data_in_pickle_file(data_, file_name)
        # Go to the next request
        _start_date = _end_date
        sleep(5.)
    except:
        print('Error: ', url)
