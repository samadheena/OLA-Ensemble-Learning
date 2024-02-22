# Input data files are available in the read-only "../input/" director
# For example, running this (by clicking run or pressing Shift+Enter)
import os
import random
#For dealing with tables
import pandas as pd
#For dealing with linear algebra
import numpy as np
#For data visualizs
_data=pd.read_csv('/Users/suraaj/Downloads/ola_driver_scaler_.csv')
_data.head()
_data=_data.drop(columns='Unnamed: 0')
_data.info()
##Converting 'MMM-YY' feature to datetime type
_data['MMM-YY'] = pd.to_datetime(_data['MMM-YY'])
##Converting 'Dateofjoining' feature to datetime type
_data['Dateofjoining'] = pd.to_datetime(_data['Dateofjoining'])
##Converting 'LastWorkingDate' feature to datetime type
_data['LastWorkingDate'] = pd.to_datetime(_data['LastWorkingDate'])
_data.info()
Step:Imputation of missing data


_data.isnull().sum()/len(_data)*100
_data['Gender'].value_counts()
_data['Education_Level'].value_counts()



KNN Imputation

_data_nums=_data.select_dtypes(np.number)
#keeping only the numerical columns
_data_nums
_data_nums.isnull().sum()
data_nums.drop(columns='Driver_ID',inplace=True)
columns=_data_nums.columns


