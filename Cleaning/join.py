import csv
import pandas as pd
import sys


def join(contracts, projects, invest_1, invest_2, outfile):
	s = "Project ID"
	print ("doing the join business")
	# df = pd.merge(contracts,on=s).merge(projects,on=s).merge(invest_1, on=s).merge(invest_2, on=s)
	df = pd.merge(contracts, invest_1, on=s)
	df.to_csv(outfile)

if __name__ == "__main__":
	contracts = pd.read_csv(sys.argv[1])
	projects = pd.read_csv(sys.argv[2])
	invest_1 = pd.read_csv(sys.argv[3])
	invest_2 = pd.read_csv(sys.argv[4])
	join(contracts, projects, invest_1, invest_2, sys.argv[5])