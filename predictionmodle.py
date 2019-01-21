#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:29:02 2017

@author: areejaltamimi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# matplotlib inline
# machine learning
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from subprocess import check_output
# Block all warning for presentation to look clean
import warnings

warnings.filterwarnings( 'ignore' )
# Print all rows and columns. Dont hide any
# pd.set_option('display.max_rows', None)
pd.set_option( 'display.max_columns', None )

# Data Files We have
import os

# filelist = os.listdir("/Users/areejaltamimi/Downloads/Big Data/Project/DATA/")
df_byyear = pd.DataFrame()
for i in range( len( filelist ) ):
    df_byyear = pd.concat(
        [df_byyear, pd.read_csv( "/Users/areejaltamimi/Downloads/Big Data/Project/DATA/" + filelist[i] )], axis=0 )

df = df_byyear
df.index = range( len( df_byyear ) )
df = pd.read_csv( "/Users/areejaltamimi/Downloads/Big Data/Project/DATA/accident_2105.csv" )
states = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas',
          6: 'California', 8: 'Colorado', 9: 'Connecticut', 10: 'Delaware',
          11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii',
          16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas',
          21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland',
          25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota',
          28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska',
          32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico',
          36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio',
          40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 43: 'Puerto Rico',
          44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee',
          48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 52: 'Virgin Islands',
          53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'}

df['state'] = df['STATE'].apply( lambda x: states[x] )
df['state'].value_counts().to_frame().transpose()
# Plotting State Wise Accident Incidents
df['state'].value_counts().to_frame().plot( kind='bar', figsize=(16, 10), title='State Wise Accident Incidents',
                                            cmap='coolwarm' )
# Drunk Driver Accident incidents
df_drunk = pd.concat( [df['state'], df['DRUNK_DR']], axis=1 )
df_drunk.groupby( 'state' ).sum().sort_index( by='DRUNK_DR', ascending=False ).plot( kind='bar', figsize=(16, 10),
                                                                                     title='State Wise Drunk Driver Accident Incidents',
                                                                                     cmap='coolwarm' )
df_nd.head()
df_nd = pd.concat( [df['state'].value_counts().to_frame(),
                    df_drunk.groupby( 'state' ).sum().sort_index( by='DRUNK_DR', ascending=False )], axis=1, )
df_nd['Accidents Because of NON Drunk Drivers'] = df_nd.state - df_nd.DRUNK_DR
df_nd.columns = ['Total Accidents', 'Accidents Because of Drunk Drivers', 'Accidents Because of NON Drunk Drivers']
df_nd.iloc[:, 1:3].plot( kind='bar', figsize=(16, 10), title='Drunk Vs Non Drunk Driver Accident', stacked=True,
                         cmap='coolwarm' )
df[['HOUR', 'MINUTE']] = df[['HOUR', 'MINUTE']].apply( lambda x: [23, 59] if int( x[0] ) == 24 else x, axis=1 )
df = df[df['DAY'] != 99]


def f(x):
    year = x[0]
    month = x[1]
    day = x[2]
    hour = x[3]
    minute = x[4]
    # Sometimes they don't know hour and minute
    if hour == 99:
        hour = 0
    if minute == 99:
        minute = 0
    s = "%02d-%02d-%02d %02d:%02d:00" % (year, month, day, hour, minute)
    c = datetime.datetime.strptime( s, '%Y-%m-%d %H:%M:%S' )
    return c


df['crashTime'] = df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']].apply( f, axis=1 )
df['crashDay'] = df['crashTime'].apply( lambda x: x.date() )
df['crashMonth'] = df['crashTime'].apply( lambda x: x.strftime( "%b" ) )
df['crashMonthN'] = df['crashTime'].apply( lambda x: x.strftime( "%d" ) )  # sorting
df['crashTime'].head()
df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']].head()
df['crashMonth'].value_counts()
fig, axes = plt.subplots( nrows=2, ncols=2, figsize=(16, 10) )
k = df['crashMonth'].value_counts()
k.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
k.plot( ax=axes[0, 0], kind='bar', title='Month Wise Accident Data', cmap='coolwarm' )

df['crashTime'].apply( lambda x: x.day ).value_counts().sort_index().plot( ax=axes[0, 1],
                                                                           title='Day Wise Accident Data',
                                                                           cmap='coolwarm' )

df['crashTime'].apply( lambda x: x.hour ).value_counts().sort_index().plot( kind='bar', ax=axes[1, 0],
                                                                            title='Hour Wise Accident Data',
                                                                            cmap='coolwarm' )

df['crashTime'].apply( lambda x: x.strftime( '%A' ) ).value_counts().plot( kind='bar', ax=axes[1, 1],
                                                                           title='Weekday Wise Accident Data',
                                                                           cmap='coolwarm' )
df['WEATHER'].value_counts()
{1: 'clear', 10: 'couldy', 2: 'rain', 5: 'fog', 4: 'snow', 99: 'unknown', 3: 'sleet', 98: 'unreported', 8: 'other',
 12: 'drizzle', 11: 'blowingSnow', 6: 'crosswinds', 7: 'blowingSand'}
f, (ax1, ax2) = plt.subplots( 1, 2, figsize=(16, 8) )
df['HARM_EV'].value_counts().head()
harm_ev = {12: 'SameRoadVehicle', 8: 'Pedestrian', 1: 'OverTurn', 42: 'Trees',
           33: 'Curb', 34: 'Ditch', 35: 'Embankment'}

df['harm_ev'] = df['HARM_EV'].apply(
    lambda x: harm_ev[x] if (x == 12 or x == 8 or x == 1 or x == 42 or x == 33 or x == 34 or x == 35)  else 'Other' )
# df['harm_ev'].value_counts().plot(kind='bar',title='Environment Playing Role in Accident',cmap='coolwarm')
sns.countplot( y=df['harm_ev'], ax=ax1 )
ax1.set_ylabel( 'Reason For Accident' )
ax1.set_title( 'Objects Playing Role in Accident' )

df['WEATHER'].value_counts()
weather = {1: 'clear', 10: 'couldy', 2: 'rain', 5: 'fog', 4: 'snow', 99: 'unknown', 3: 'sleet', 98: 'unreported',
           8: 'other', 9: 'other', 12: 'drizzle', 11: 'blowingSnow', 6: 'crosswinds', 7: 'blowingSand'}

df['weather'] = df['WEATHER'].apply( lambda x: weather[x] )
sns.countplot( y=df['weather'], ax=ax2 )
plt.ylabel( 'Weather when accident Happened' )
plt.title( 'Weather When Accident Happened' )
# df['weather'].value_counts().plot.bar(figsize=(8,4))
df.columns
df['crashTime'].head()
rng = pd.date_range( '1/1/2015', '31/12/2015', freq='1H' )
ts = pd.Series( np.random.randint( 0, 5, len( rng ) ), index=rng )
ts.head()
df['crashTime'][0]
df['crashTime'].dt.day[0]
ts_count.head()
datetime.datetime
ts_count = pd.DataFrame(
    m['crashTime'].apply( lambda x: x.replace( microsecond=0, second=0, minute=0 ) ).value_counts(), index=rng )

ts_count.columns = ['Crash_no']

ts_count['Crash_no'].fillna( 0 )
ts_count.reset_index( inplace=True )
ts_count.columns
ts_count['Day'] = ts_count['index'].apply( lambda x: x.day )
ts_count['Month'] = ts_count['index'].apply( lambda x: x.strftime( "%b" ) )
ts_count['Hour'] = ts_count['index'].apply( lambda x: x.hour )
ts_count['Weekday'] = ts_count['index'].apply( lambda x: x.strftime( '%A' ) )
rng = pd.date_range( '1/1/2007', '31/12/2015', freq='1H' )
df_day = pd.DataFrame( df['crashTime'].apply( lambda x: x.replace( microsecond=0, second=0, minute=0 ) ).value_counts(),
                       index=rng )

df_day.columns = ['Crash_no']

df_day['Crash_no'].fillna( 0 )
df_day.reset_index( inplace=True )
df_day.columns
df_day['Day'] = df_day['index'].apply( lambda x: x.day )
df_day['Month'] = df_day['index'].apply( lambda x: x.strftime( "%b" ) )
df_day['Hour'] = df_day['index'].apply( lambda x: x.hour )
df_day['Weekday'] = df_day['index'].apply( lambda x: x.strftime( '%A' ) )
df_day['Year'] = df_day['index'].apply( lambda x: x.strftime( '%G' ) )
df_stats = df_day
df_stats['Crash_no'] = df_stats['Crash_no'].fillna( 0 )

df_stats['Hour'] = df_stats['Hour'].apply( str )
df_stats['Day'] = df_stats['Day'].apply( str )
df_stats['Year'] = df_stats['Year'].apply( str )
import statsmodels.formula.api as smf

mod = smf.ols( formula='Crash_no ~ Year+ Hour+ Day+ Month+Weekday', data=df_stats )
res = mod.fit()
print( res.summary() )
ts_count.head()
df_ml = ts_count[['Day', 'Month', 'Hour', 'Weekday', 'Crash_no']]
ts_count.columns
df_ml.head()
rng = pd.date_range( '1/1/2007', '31/12/2015', freq='1D' )
df_day = pd.DataFrame(
    df['crashTime'].apply( lambda x: x.replace( microsecond=0, second=0, minute=0, hour=0 ) ).value_counts(),
    index=rng )

df_day.columns = ['Crash_no']

df_day['Crash_no'].fillna( 0 )
df_day.reset_index( inplace=True )
df_day.columns
df_day['Day'] = df_day['index'].apply( lambda x: x.day )
df_day['Month'] = df_day['index'].apply( lambda x: x.strftime( "%b" ) )
df_day['Hour'] = df_day['index'].apply( lambda x: x.hour )
df_day['Weekday'] = df_day['index'].apply( lambda x: x.strftime( '%A' ) )
df_day['Year'] = df_day['index'].apply( lambda x: x.strftime( '%G' ) )

df_stats = df_day
df_stats['Crash_no'] = df_stats['Crash_no'].fillna( 0 )

df_stats['Hour'] = df_stats['Hour'].apply( str )
df_stats['Day'] = df_stats['Day'].apply( str )
df_stats['Year'] = df_stats['Year'].apply( str )

import statsmodels.formula.api as smf

mod = smf.ols( formula='Crash_no ~ Day+ Month+Weekday+Year', data=df_stats )
res = mod.fit()
print( res.summary() )
mod = smf.ols( formula='Crash_no ~ Month+Weekday+Year', data=df_stats )
res = mod.fit()
print( res.summary() )
x = pd.concat( [pd.get_dummies( df_day['Day'], prefix='Day', drop_first=True ),
                pd.get_dummies( df_day['Month'], prefix='Month', drop_first=True ),
                pd.get_dummies( df_day['Weekday'], prefix='Weekday', drop_first=True ),
                pd.get_dummies( df_day['Year'], prefix='Year', drop_first=True )], axis=1 )
Y = df_day['Crash_no'].fillna( 0 )
x.head()
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split( x, Y, test_size=0.3, random_state=0 )

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
y_train = sc.fit_transform( y_train )
y_test = sc.transform( y_test )
from scipy.stats import norm

sns.distplot( Y, fit=norm );
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Validation function
n_folds = 5


def rmsle_cv(model):
    kf = KFold( n_folds, shuffle=True, random_state=42 ).get_n_splits( X_train.values )
    rmse = np.sqrt( -cross_val_score( model, X_train.values, y_train, scoring="neg_mean_squared_error", cv=kf ) )
    return (rmse)


from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lasso = make_pipeline( RobustScaler(), Lasso( alpha=0.0005, random_state=1 ) )

ENet = make_pipeline( RobustScaler(), ElasticNet( alpha=0.0005, l1_ratio=.9, random_state=3 ) )

KRR = KernelRidge( alpha=0.6, kernel='polynomial', degree=2, coef0=2.5 )

GBoost = GradientBoostingRegressor( n_estimators=3000, learning_rate=0.05,
                                    max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10,
                                    loss='huber', random_state=5 )

model_xgb = xgb.XGBRegressor( colsample_bytree=0.4603, gamma=0.0468,
                              learning_rate=0.05, max_depth=3,
                              min_child_weight=1.7817, n_estimators=2200,
                              reg_alpha=0.4640, reg_lambda=0.8571,
                              subsample=0.5213, silent=1,
                              random_state=7, nthread=-1 )

model_lgb = lgb.LGBMRegressor( objective='regression', num_leaves=5,
                               learning_rate=0.05, n_estimators=720,
                               max_bin=55, bagging_fraction=0.8,
                               bagging_freq=5, feature_fraction=0.2319,
                               feature_fraction_seed=9, bagging_seed=9,
                               min_data_in_leaf=6, min_sum_hessian_in_leaf=11 )
models = [lm, lasso, ENet, KRR, GBoost, model_xgb, model_lgb]
model_name = ['Multiple Linear Regression', 'Lasso Regression', 'Elastic Net', 'Kernel Ridge', 'Gradient Boosting',
              'Xg Boost', 'Light Gradient Boosting']
model_score = pd.DataFrame( columns=['Name', 'Mean_Rmse', 'Std_dev'] )
for i in range( len( models ) ):
    model_score.loc[i, 'Name'] = model_name[i]
    model_score.loc[i, 'Mean_Rmse'] = rmsle_cv( models[i] ).mean()
    model_score.loc[i, 'Std_dev'] = rmsle_cv( models[i] ).std()

model_score
sns.barplot( y='Name', x='Mean_Rmse', data=model_score )
plt.xlim( 0, 1 );
plt.xlabel( 'Root Mean Sq Error' )
for i in range( len( models ) ):
    models[0].fit( X_train, y_train )
    predictions = models[0].predict( X_test )
    fig, axes = plt.subplots( nrows=1, ncols=2, figsize=(16, 6) )
    axes[0].scatter( y_test, predictions )
    axes[0].set_title( 'Prediction Vs Test Data Plot' )
    plt.legend()

    sns.distplot( (y_test - predictions), bins=30, ax=axes[1] );
    axes[1].set_title( 'Residual Histogram' )
    plt.suptitle( 'Evaluation Of ' + model_name[i] + ' Model Prediction' )
