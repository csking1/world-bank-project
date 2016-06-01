import csv
import pandas as pd
import sys

# fix this join, it's not turning out the right things

def join(contracts, projects, invest_1, invest_2, outfile):
	s = "Project ID"
	print ("merging 1 / 3")
	a = pd.concat([invest_1, invest_2])
	print ("merging 2 / 3")
	b = pd.merge(a, projects, on = s)
	print ("merging 3 / 3")
	df = pd.merge(contracts, b, on=s)
	print ("dropping duplicate entries")
	df = df.drop_duplicates([s], keep = "last" )
	print ("writing to csv")
	df.to_csv(outfile)

if __name__ == "__main__":
	contracts = pd.read_csv(sys.argv[1])
	projects = pd.read_csv(sys.argv[2])
	invest_1 = pd.read_csv(sys.argv[3])
	invest_2 = pd.read_csv(sys.argv[4])
	join(contracts, projects, invest_1, invest_2, sys.argv[5])