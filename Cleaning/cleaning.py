import csv
import pandas as pd
import sys

def clean(df, outfile):
	'''
	Takes a pandas data frame, replaces null values, and writes out a cleaned version to the csv
	'''
	new = df.fillna(df.mode().iloc[0])
	print (new.isnull().sum())
	df.to_csv(outfile)

if __name__ == "__main__":
	df = pd.read_csv(sys.argv[1])
	clean(df, sys.argv[2])