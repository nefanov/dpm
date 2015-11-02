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
import rpy2

from rpy2 import *
import rpy2.robjects as RO


def prepare_csv(fd="raw.txt",cpus=2):
	tim=list()
	val=list()
	f = open(fd,'r')
	for line in f:
		if (line[:5]=="DATE:"):
			for i in range(cpus):
				tim.append(line[6:-1])
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
#R0.r('y =scan(file.choose())')
# use example WWWusage data
	RO.r('fit <- auto.arima(dat)')
	return


T,V = prepare_csv('sysbench_run.log')
save_csv(T,V)
forecasting_arima()
