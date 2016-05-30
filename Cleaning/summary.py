import csv
import pandas as pd
import sys

def summary(df):
	'''
	Takes a pandas data frame, generates summary statistics, and writes out a cleaned version of the csv
	'''
	print ("Mode:", df.mode())

	# for i in range(df.shape[1]):
	# 	print df.iloc(i)

	# find nans and replace
	# get correlations of columns with each other
	# get
	# write to csv

def projects(df):
	'''
	Takes a data frame, generates summary statistics, and writes out a cleaned version of the csv
	'''
	print (df.shape)


def investigations(df):
	'''
	Takes a data frame, generates summary statistics, and writes out a cleaned version of the csv
	'''
	print (df.shape)



if __name__ == "__main__":
	take = pd.read_csv(sys.argv[1])
	summary(take)
	# if "contracts" in sys.argv[1]:
	# 	contracts(take)
	# elif "projects" in sys.argv[1]:
	# 	projects(take)
	# elif "investigations" in sys.argv[1]:
	# 	investigations(take)