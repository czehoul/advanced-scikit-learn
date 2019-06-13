#This example show some advanced Scikit-Learn transformer constructs like Function Transformer,
#Feature Union, Column Transformer and how to stack the transformers in Pipeline in sequence  
#
from CustomTransformer import round_time, Cat2VecTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler

#Function transformer - use for stateless transformation
round_time_transformer = FunctionTransformer(round_time, validate=False)

#Feature union - imputation + masking
impute_and_mask = FeatureUnion(
  transformer_list=[
      ('features', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
      ('indicators', MissingIndicator())])


cat_cols = ['ip', 'app', 'device', 'os', 'channel']
impute_and_mask_cols = ['count_by_ip_HOUR', 'count_by_ip_app', 'count_by_ip_app_os',
       'countunique_channel_by_ip', 'countunique_app_by_ip_device_os',
       'countunique_app_by_ip', 'countunique_os_by_ip_app',
       'countunique_device_by_ip', 'countunique_channel_by_app',
       'cumcount_by_ip', 'cumcount_by_ip_device_os',
       'last_click_by_ip_app_device_os_channel_DESC',
       'last_click_by_ip_os_device_DESC',
       'last_click_by_ip_os_device_app_DESC',
       'last_click_by_ip_channel_DESC',
       'last_click_by_ip_app_device_os_channel_ASC',
       'last_click_by_ip_os_device_ASC',
       'last_click_by_ip_os_device_app_ASC',
       'last_click_by_ip_channel_ASC']

#Use of custom transformer
cat2vecTransformer = Cat2VecTransformer(windows=5, min_count=5, cat_feature_size=5)

#Column Transformer - to select specified columns as input to a transformer
preprocess_cols = make_column_transformer(
   (round_time_transformer, ['click_time']) ,
    (impute_and_mask, impute_and_mask_cols) ,
    (cat2vec, cat_cols)       
)

scaler = StandardScaler()
#Stacking the transfomer the they are process in sequence
all_preprocess =  Pipeline([('preprocess', preprocess_cols), ('scaler', scaler)])
