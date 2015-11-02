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


def prepare_csv(fd="raw.txt"):
	f = open(fd,'r')
	for line in f:
		#do smth
		val.append()
		tim.append()
	return tim,val
	


def forecasting_arima(csvname="data.csv"):
	RO.r('library(forecast)')
	RO.r('dat = read.csv("'+csvname+'", header = TRUE)')
#R0.r('y =scan(file.choose())')
# use example WWWusage data
	RO.r('fit <- auto.arima(dat)')

forecasting_arima("data.csv")
