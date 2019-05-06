import pytest 
import pandas as pd
import numpy as np 


@pytest.mark.functions
def test_change_type(dataframe):
    df = dataframe.change_type(column="a", dtype=float)
    assert df["a"].dtype == float
    
    

@pytest.mark.functions
def test_change_type_keep_values():
    df=pd.DataFrame(['a',1,True], columns=['col1'])
    df = df.change_type(column="col1", dtype=float, ignore_exception="keep_values")
    assert df.equals(pd.DataFrame(['a',1,True], columns=['col1']))
    
    
@pytest.mark.functions
def test_change_type_fillna():
    df=pd.DataFrame(['a',1,True], columns=['col1'])
    df = df.change_type(column="col1", dtype=float, ignore_exception="fillna")
    assert np.isnan(df.col1[0])

    
@pytest.mark.functions
def test_change_type_unknown_option():
    df=pd.DataFrame(['a',1,True], columns=['col1'])
    with pytest.raises(Exception):
        df = df.change_type(column="col1", dtype=float, ignore_exception="blabla")
        
        
@pytest.mark.functions
def test_change_type_raise_exception():
    df=pd.DataFrame(['a',1,True], columns=['col1'])
    with pytest.raises(Exception):
        df = df.change_type(column="col1", dtype=float, ignore_exception=False)