import csv
import pandas as pd
import sys

# fix this join, it's not turning out the right things

def join(contracts, projects, invest_1, outfile):
	s = "Project ID"
	# print ("merging 1 / 3")
	# a = pd.concat([invest_1, invest_2])
	print ("merging 1 / 2")
	b = pd.merge(invest_1, projects, on = s)
	print ("merging 2 / 2")

	# maybe try a left join here?
	df = pd.merge(contracts, b, on=s)

	print ("dropping duplicate entries")
	try:
		df = df.drop_duplicates([s], take_last = True )
	except:
		print ("couldn't drop duplicates")
	print ("writing to csv")
	df.to_csv(outfile)

if __name__ == "__main__":
	contracts = pd.read_csv(sys.argv[1])
	projects = pd.read_csv(sys.argv[2])
	invest_1 = pd.read_csv(sys.argv[3])
	join(contracts, projects, invest_1, sys.argv[4])