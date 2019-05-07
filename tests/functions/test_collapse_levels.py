import pandas as pd
import pytest


@pytest.mark.functions
def test_collapse_levels_sanity(multiindex_with_missing_dataframe):
    with pytest.raises(TypeError):
        multiindex_with_missing_dataframe.collapse_levels(sep=3)


@pytest.mark.functions
def test_collapse_levels_non_multilevel(multiindex_with_missing_dataframe):
    # an already single-level DataFrame is not distorted
    pd.testing.assert_frame_equal(
        multiindex_with_missing_dataframe.copy().collapse_levels(),
        multiindex_with_missing_dataframe.collapse_levels().collapse_levels(),
    )


@pytest.mark.functions
def test_collapse_levels_functionality_2level(
    multiindex_with_missing_dataframe
):

    assert all(
        multiindex_with_missing_dataframe.copy()
        .collapse_levels()
        .columns.values
        == ["a", "Normal  Distribution", "decorated-elephant_r.i.p-rhino :'("]
    )
    assert all(
        multiindex_with_missing_dataframe.copy()
        .collapse_levels(sep="AsDf")
        .columns.values
        == [
            "a",
            "Normal  Distribution",
            "decorated-elephantAsDfr.i.p-rhino :'(",
        ]
    )


    
@pytest.mark.functions
def test_collapse_levels_functionality_3level(
    multiindex_with_missing_3level_dataframe
):
    assert all(
        multiindex_with_missing_3level_dataframe.copy()
        .collapse_levels()
        .columns.values
        == [
            "a",
            "Normal  Distribution_Hypercuboid (???)",
            "decorated-elephant_r.i.p-rhino :'(_deadly__flamingo",
        ]
    )
    assert all(
        multiindex_with_missing_3level_dataframe.copy()
        .collapse_levels(sep="AsDf")
        .columns.values
        == [
            "a",
            "Normal  DistributionAsDfHypercuboid (???)",
            "decorated-elephantAsDfr.i.p-rhino :'(AsDfdeadly__flamingo",
        ]
    )
    
@pytest.mark.functions
def test_collapse_levels_multilevel_index():
    df= pd.DataFrame([['a',1],['a',2],['b',3]], columns=['col1','col2'], index = pd.MultiIndex.from_tuples([(0,0),(0,1),(1,0)], names=['i1','i2'])) 
    df1= df.collapse_levels(sep='-', axis=0)
    df2= pd.DataFrame([['a',1],['a',2],['b',3]],index=['0-0','0-1','1-0'], columns=['col1','col2'])
    assert (df1.index==df2.index).all() and (df1==df2).all().all()
    
   



