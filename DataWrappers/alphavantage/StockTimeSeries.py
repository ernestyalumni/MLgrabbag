"""
@file StockTimeSeries.py
@ref https://www.alphavantage.co/documentation/
@ref https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/timeseries.py

@details Based upon RomelTorres' alpha_vantage github repository code.
"""

"""
@ref https://www.python-course.eu/python3_decorators.php
@brief A primer on Python decorators.

@details Python has 2 different kinds of decorators:
* Function decorators
* Class decorators

Due to the fact that every parameter of a function is a reference to an object
and functions are objects as well, we can pass functions - or more precisely
"references to functions" - as parameters to a function.

The output of a function is also a reference to an object. Therefore, functions
can return references to function objects.
"""
from DataWrappers.utilities.Configuration import Configuration
from DataWrappers.utilities.handle_API_call import (handle_API_call,
  response_from_API_call)

from collections import namedtuple
from functools import wraps
from urllib.parse import urlencode

# Low-level API wrappers

TimeSeriesAPIParameters = \
    namedtuple('TimeSeriesAPIParameters', \
               ['function', \
                'symbol', \
                'interval', \
                'outputsize', \
                'datatype', \
                'apikey'])

def time_series_intraday(symbol, \
                         interval='1min', \
                         outputsize='compact', \
                         datatype='json'):
    """
    @fn time_series_intraday
    @brief Low-level API wrapper for the Alpha Vantage API

    @param symbol Required API parameter for the name of the equity of your
    choice, e.g. symbol=MSFT
    @param interval Required API parameter for the time interval between 2
    consecutive data points in the time series.
    """
    supported_time_series_values = ['1min', '5min', '15min', '30min', '60min']
    if interval not in supported_time_series_values:
        raise IOError(interval, " is not a supported interval.")
        
    supported_output_sizes = ['compact', 'full']
    if outputsize not in supported_output_sizes:
        raise IOError(outputsize, " is not a supported outputsize.")
        
    supported_datatypes = ['json', 'csv']
    if datatype not in supported_datatypes:
        raise IOError(datatype, " is not a supported datatype.")
    
    configuration = Configuration()
    API_key = \
        configuration.get_configparser()["Authentication"]\
            ['alphavantage_API_key']    
    
    return TimeSeriesAPIParameters('TIME_SERIES_INTRADAY', \
                                   symbol, \
                                   interval, \
                                   outputsize, \
                                   datatype, \
                                   API_key)

def url_query_string_from_API_parameters(API_parameters):
    return urlencode(API_parameters._asdict())

def _to_url(function):
    configuration = Configuration()
    API_url = \
        configuration.get_configparser()["Paths"]['Alpha_Vantage_API_url']    
    
    @wraps(function)
    def function_wrapper(*args, **kwargs):
        
        API_parameters = function(*args, **kwargs)

        query_string = url_query_string_from_API_parameters(API_parameters)

        return API_url + query_string

    return function_wrapper

@_to_url
def url_for_time_series_intraday(symbol, \
                                 interval='1min',
                                 outputsize='compact',
                                 datatypes='json'):
    return time_series_intraday(symbol, interval, outputsize, datatypes)


def _request_from_url(function):
    
    @wraps(function)
    def function_wrapper(*args, **kwargs):
        
        url = function(*args, **kwargs)
        
        return handle_API_call(url)

    return function_wrapper

@_request_from_url
def url_request_for_time_series_intraday(symbol, \
                                         interval='1min',
                                         outputsize='compact',
                                         datatypes='json'):
    return url_for_time_series_intraday(symbol, \
                                        interval, \
                                        outputsize, \
                                        datatypes)

def _open_with_url(function):
    
    @wraps(function)
    def function_wrapper(*args, **kwargs):
        
        url = function(*args, **kwargs)
        
        return response_from_API_call(url)

    return function_wrapper

@_open_with_url
def url_response_for_time_series_intraday(symbol, \
                                          interval='1min',
                                          outputsize='compact',
                                          datatypes='json'):
    """
    @fn url_response_for_time_series_intraday
    
    @param symbol The symbol for the equity we want to get data about.
    @param interval Time interval between 2 consecutive values.
    Supported values are
        '1min'
        '5min'
        '15min'
        '30min'
        '60min'
        (default is '1min')
    @param output_size The size of the call. The desired size for the output.
    Supported values are 'compact' and 'full'.
    'compact' returns the last 100 points in the data series, and
    'full' returns the full-length intraday time series, commonly above 1MB.
    Default is 'full'
    """
    return url_for_time_series_intraday(symbol, \
                                        interval, \
                                        outputsize, \
                                        datatypes)
        

#@AlphaVantage._output_format
#@AlphaVantage._call_API_on_function
#def load_intraday(self, symbol, interval='1min', output_size='full')
"""
@fn load_intraday
@brief Return time series in two JSON as data and metadata.
@details Can raise ValueError.
Based upon RomelTorres' implementation timeseries.py in alpha_vantage github
repository.

@param symbol The symbol for the equity we want to get data about.
@param interval Time interval between 2 consecutive values.
Supported values are
    '1min'
    '5min'
    '15min'
    '30min'
    '60min'
    (default is '1min')
@param output_size The size of the call. The desired size for the output.
Supported values are 'compact' and 'full'.
'compact' returns the last 100 points in the data series, and
'full' returns the full-length intraday time series, commonly above 1MB.
Default is 'full'
"""
#  _FUNCTION_KEY = "TIME_SERIES_INTRADAY"

 # return _FUNCTION_KEY, "Time Series ({})".format(interval), 'Metadata'






