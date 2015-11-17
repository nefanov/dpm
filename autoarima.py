# configure the environment - because of my local machine options - nevermind
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
# program:
import readline
#import pyplot as plt
import rpy2.robjects
import scipy as sp
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame, Series
#from pandas import time_range
import matplotlib.pyplot as plt
import statsmodels.api as sm
import rpy2
import matplotlib
from datetime import datetime
from statsmodels.iolib.table import SimpleTable
from rpy2 import *
import rpy2.robjects as RO
from sklearn.metrics import r2_score
import ml_metrics as metrics

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
	return tim[::2],val[::2]

def save_csv(tim,val,fd='data.csv'):
	f=open(fd,'w')
	f.write('datetime;J\n')
	for i in range(len(tim)):
		f.write(tim[i]+';'+val[i]+'\n')
	f.close()
	return


def forecasting_arima(csvname="data.csv"):
	RO.r('library(forecast)')
	RO.r('dat = read.csv("'+csvname+'", header = TRUE)')
	RO.r('fit <- auto.arima(dat,trace=TRUE,allowdrift=TRUE)')
	
	#return numpy.array(RO.r(fit))


T,V = prepare_csv('sysbench_run.log')
#then save data
save_csv(T,V)
#forecasting_arima()
#do_analysys

dataset = read_csv('data.csv', ';', index_col=['datetime'],parse_dates=['datetime'])

dataset.head()

print "DATASET:"

otg = dataset.J
otg.head()
x=list()
y=list()
y=np.array(otg.values).tolist()

for i in range(len(otg.values)):
	otg.values[i] = y[i]
#filter misaligned values because of RAPL overflow(as int64) by average of neighbours:
for i in range(len(otg.index)-1):
        if (otg.values[i] < 0.0):
                if (i==0):
                        otg.values[i] = otg.values[i+1]/2;
                else:
                        otg.values[i] = (otg.values[i-1] + otg.values[i+1])/2;
print 'res'
'''
plt.grid()
plt.plot(otg.index,otg.values)
plt.show()
'''
otg=otg['2015-11-08 18:53:03':]

plt.grid()
plt.plot(otg.index,otg.values)
plt.show()

print otg
otg.to_csv("filtered.csv")
#forecasting_arima("filtered.csv")

#itog=otg.describe()

#itog

#print 'V = %f' % (itog['std']/itog['mean'])

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
src_data_model = otg[:'2015-11-09 03:26:26']
model = sm.tsa.ARIMA(src_data_model, order=(4,1,0)).fit() #trend='nc' if need
print model.summary()
q_test = sm.tsa.stattools.acf(model.resid, qstat=True)
print DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]})

pred = model.predict('2015-11-09 03:26:16','2015-11-09 03:29:06', typ='levels')
trn = otg['2015-11-09 03:26:26':]
r2 = r2_score(trn, pred)
print 'R^2: %1.2f' % r2
#mean-square rmse

metrics.rmse(trn,pred)

metrics.mae(trn,pred)

fig, (ax1) = plt.subplots(nrows = 1, sharex=True)
ax1.plot(otg.index,otg.values)

ax1.plot_date(pred.index,pred.values,'r--')

plt.show()

#print pred.values
