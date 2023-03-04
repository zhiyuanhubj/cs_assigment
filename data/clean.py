
import pandas as pd

df_condo_dirty = pd.read_csv('a1-condo-resale-dirty.csv')

df_cleaned = df_condo_dirty.copy()


#df_cleaned.drop(labels=df_cleaned['area_sqft']<0,axis=1, inplace=True)

# area_sqft attribute should be a positive integer value, while there are some records with the '-1' value in 'area_sqft' attribute.


########## 1(b)

df_condo_nan = pd.read_csv('a1-condo-resale-nan.csv')

df_no_nan = df_condo_nan.copy()

df_condo_nan.drop(columns=['url'])

df_condo_nan.dropna(axis=0,subset=['price','area_sqft']) # Remove all records where price or area_sqft is NaN

print(df_condo_nan.head())

df_condo_nan.info()


#construct the mapping between planning_area and subzone

dict_subzone_area = df_condo_nan.set_index(['subzone'])['planning_area'].to_dict()

print(dict_subzone_area)




########## 1(c)



