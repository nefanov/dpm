# configure the environment - because of my local machine options - nevermind
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
# program:
import readline
import rpy2.robjects
import scipy as sp
import numpy
import pandas
from pandas import read_csv, DataFrame, Series
from pandas import time_range
import statsmodels.api as sm
import rpy2

from rpy2 import *
import rpy2.robjects as RO


def prepare_csv(fd="raw.txt",cpus=2):
	tim=list()
	val=list()
	f = open(fd,'r')
	for line in f:
		if (line[:4]=="2015"):
			for i in range(cpus):
				tim.append(line[:-1])
		if (line.find("value=")>-1):
			val.append(line[ line.find("value=")+6 : line.find("J") ] )
	f.close()	
	return tim,val

def save_csv(tim,val,fd='data.csv'):
	f=open(fd,'w')
	f.write('time J\n')
	for i in range(len(tim)):
		f.write(tim[i]+' '+val[i]+'\n')
	f.close()
	return


def forecasting_arima(csvname="data.csv"):
	RO.r('library(forecast)')
	RO.r('dat = read.csv("'+csvname+'", header = TRUE)')
	RO.r('fit <- auto.arima(dat,trace=TRUE,allowdrift=TRUE)')
	
	return numpy.array(RO.r(fit))


T,V = prepare_csv('sysbench_run.log')
save_csv(T,V)
forecasting_arima()
#go

#otg = sample.resample('5Min', how=conversion, base=30)
dataset = read_csv('data.csv')
dataset.head()

otg=dataset.resample('60Min', how=conversion, base=30)
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

src_data_model = otg[:'2013-05-26']
model = sm.tsa.ARIMA(src_data_model, order=(3,1,1), freq='10S').fit(full_output=False, disp=0)
