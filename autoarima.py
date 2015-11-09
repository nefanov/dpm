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
print forecasting_arima()
