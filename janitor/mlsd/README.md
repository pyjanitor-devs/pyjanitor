#pre-processing
Directory contains any janitor/mlsd (machine learning structued data) data pre-processing functions:

- (3/20/2021) port to pyjanitor package
    - fork bcottman/pyjanitor
    - clone local project
    - create and test and use ../janitor/docker
    - create and test and use ../devops.py
    - create local ../pyjanitor/ janitor/mlsd
    - mdir ../janitor/mlsd (ml structured data)
    - mdir ../tests/mlsd (ml structured data)
    - all tests @pytest.mark.mlsd (mlsd)
    - port utils.py and test_utils.py
    - port todatatpe.py and test_todatatpe

- (6/6/2020) All functions accept numpy arrays only,
- Every array is 2-D. 
- Vertical index (n) is row index
- Horizontal index (m) is column index


- (3/9/19) EliminateUnviableFeatures:
    - Elimnate Features not found in train and test
    - Eliminate duplicate features ; check all values
    - Eliminate single unique value features
    - Eliminate low variance features:  > std/(max/min)

- (3/6/19) toCategory : converts any type to a category integer;
- (3/6/19) ContinuoustoCategory :
- (3/7/19) toDateTimeComponents:

- (3/12,13,14,15) clean doc;  docstring; code; git push,added a lot of new text
- (3/20/19) eliminated /common , moved files into base.py
- (3/20/19) read_paso_parameters TEST, PLUS RUN ALL OTHER TESTS
- (3/11/19) clean doc
- (3/20/19) scale class
- (3/21/19) scale added dask_pandas_ratio and docstring and test
- (3/22,23/19) created and tested pipelines, pydot paso DAG
- (3/24/19) added tests to base, got rid of DateTimetoCategory
- (3/25/19) create GITHUB ACCOUNT
- (36-3/28/19) Sick
- (3/29-4/2/19) decorator TransformWrap to ease creating a functionTransform
- (4/6-7/19) decorator pasoDecorators. refactor,test
- (4/11/2019) test 
- (4/13-15/2019) package pypi, pip installreadthedocs
- (4/16/2019) pypi, readthedocs debugging
- () lesson 1
- () medium article 1


- (TBD)gaussinize: transform distribution as much as possible to gaussian form; 
- - () add Yeo
- - () time box cox vs scikit version
- () clean doc;  docstring; code; git push 
- () lesson 2
= () medium article 

- (TBD)impute: fill in missing data; use missingno package
- () clean doc;  docstring; code; git push 
- () lesson 3
= () medium article 

- (TBD)toEmbedded: converts any type to a category and then to embedded encoding;
- () clean doc;  docstring; code; git push 
- () lesson 4
= () medium article 

- (TBD)cleanText: currenllty
    - - .drop_punctuation
    - - .drop_newline 
    - - .drop_multispaces 
    - - .all_lower_case 
    - - .fill_na_with 
    - - .deduplication_threshold 
    - - .apostrophes 
    - - .use_stopwords

- (TBD)balance: under-sample over-sample a class;




- (TBD)augment: currently only 2d image data;

- (TBD)models (use param to control hypertune parameters)
- - rf
- - xgboost
- - lightgbm
- - catboost 


