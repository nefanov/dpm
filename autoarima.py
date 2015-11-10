# configure the environment - because of my local machine options - nevermind
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
# program:
import readline
#import pyplot as plt
import rpy2.robjects
import scipy as sp
import numpy
import pandas as pd
from pandas import read_csv, DataFrame, Series
#from pandas import time_range
import matplotlib.pyplot as plt
import statsmodels.api as sm
import rpy2
from statsmodels.iolib.table import SimpleTable
from rpy2 import *
import rpy2.robjects as RO
from sklearn.metrics import r2_score

def prepare_csv(fd="raw.txt",cpus=2):
	tim=list()
	val=list()
	f = open(fd,'r')
	for line in f:
		if (line[:2]=="T:"):
			for i in range(cpus):
				tim.append(line[2:-1])
		if (line.find("value=")>-1):
			val.append(line[ line.find("value=")+6 : line.find("J") ] )
	f.close()	
	
	#print tim
	return tim[::2],val[::2]

def save_csv(tim,val,fd='data.csv'):
	f=open(fd,'w')
	f.write('date time J\n')
	for i in range(len(tim)):
		f.write(tim[i]+' '+val[i]+'\n')
	f.close()
	return


def forecasting_arima(csvname="data.csv"):
	RO.r('library(forecast)')
	RO.r('dat = read.csv("'+csvname+'", header = TRUE)')
	RO.r('fit <- auto.arima(dat,trace=TRUE,allowdrift=TRUE)')
	
	#return numpy.array(RO.r(fit))


T,V = prepare_csv('sysbench_run.log')
save_csv(T,V)
forecasting_arima()
#go

dataset = read_csv('data.csv', ' ')

print dataset
dataset.head()
for i in range(len(dataset.date.values)):
	dataset.date.values[i] += ' ' +  dataset.time.values[i]
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.index=dataset.date.values
dataset=dataset.drop(['date','time'],axis=1)
print "DATASET:"
print dataset

otg = dataset.J
otg.head()
itog=otg.describe()
otg.hist()
itog

print 'V = %f' % (itog['std']/itog['mean'])

otg1diff = otg.diff(periods=1).dropna()
m = otg1diff.index[len(otg1diff.index)/2+1]
r1 = sm.stats.DescrStatsW(otg1diff[m:])
r2 = sm.stats.DescrStatsW(otg1diff[:m])
print 'p-value: ', sm.stats.CompareMeans(r1,r2).ttest_ind()[1]

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(otg1diff.values.squeeze(), lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(otg1diff, lags=25, ax=ax2)
print 'otg'
print otg
src_data_model = otg
print "SRC"
print src_data_model
model = sm.tsa.ARIMA(src_data_model, order=(4,1,0)).fit(trend='nc')
print model.summary()
q_test = sm.tsa.stattools.acf(model.resid, qstat=True)
print DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]})

pred = model.predict('2015-11-09 03:26:26','2015-11-09 03:29:06', typ='levels')
trn = otg['2015-11-09 03:26:26':]
r2 = r2_score(trn, pred[1:32])
print 'R^2: %1.2f' % r2
