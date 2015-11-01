# configure the environment - because of my local machine options - nevermind
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
# program:
import scipy as sp
import numpy
import pandas
import rpy2

from rpy2 import *
import rpy2.robjects as RO

RO.r('library(forecast)')

# use example WWWusage data
RO.r('fit <- auto.arima(WWWusage)')

