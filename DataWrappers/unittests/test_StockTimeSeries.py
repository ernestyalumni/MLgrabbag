# -*- coding: utf-8 -*-
"""
@file test_Configuration.py
"""
import pytest

from DataWrappers.alphavantage.StockTimeSeries import (time_series_intraday,
  url_query_string_from_API_parameters,
  url_for_time_series_intraday)

@pytest.fixture
def alpha_vantage_API_key():
    from DataWrappers.utilities.Configuration import Configuration
    configuration = Configuration()
    return configuration.get_configparser()["Authentication"]\
        ['alphavantage_API_key']   

@pytest.fixture
def alpha_vantage_API_url():
    from DataWrappers.utilities.Configuration import Configuration
    configuration = Configuration()
    return configuration.get_configparser()["Paths"]['Alpha_Vantage_API_url']   


def test_time_series_intraday_default_arguments_works():
    """
    @fn test_time_series_intraday_works
    """
    test_symbol = "ABCDE"
    result = time_series_intraday(test_symbol)

    assert 'function' in result._asdict()
    assert 'symbol' in result._asdict()
    assert 'interval' in result._asdict()
    assert 'outputsize' in result._asdict()
    assert 'datatype' in result._asdict()
    assert 'apikey' in result._asdict()
    
    assert result.function == 'TIME_SERIES_INTRADAY'
    assert result.symbol == test_symbol
    assert result.interval == '1min'
    assert result.outputsize == 'compact'
    assert result.datatype == 'json'
    
def test_url_query_string_from_API_parameters(alpha_vantage_API_key):
    test_symbol = "ABCDE"
    api_parameters = time_series_intraday(test_symbol)
    
    result = url_query_string_from_API_parameters(api_parameters)

    expected_results_part_1 = \
        "function=TIME_SERIES_INTRADAY&symbol=ABCDE&interval=1min&outputsize=compact&datatype=json"
        
    expected_results_part_2 = "&apikey=" + alpha_vantage_API_key
    
    assert result == (expected_results_part_1 + expected_results_part_2)
    
def test_url_for_time_series_intraday(alpha_vantage_API_key, \
                                      alpha_vantage_API_url):
    test_symbol = "ABCDE"
    result = url_for_time_series_intraday(test_symbol)

    expected_results_part_2 = \
        "function=TIME_SERIES_INTRADAY&symbol=ABCDE&interval=1min&outputsize=compact&datatype=json"
        
    expected_results_part_3 = "&apikey=" + alpha_vantage_API_key

    assert result == alpha_vantage_API_url + expected_results_part_2 + \
        expected_results_part_3
