import csv
import pandas as pd
import sys



def contracts(df):
	'''
	Takes a pandas data frame and generates summary statistics and charts
	'''
	print (df.shape)












if __name__ == "__main__":
	take = pd.read_csv(sys.argv[1])
	if "contracts" in sys.argv[1]:
		contracts(take)
	elif "projects" in sys.argv[1]:
		projects(take)
	elif "investigations" in sys.argv[1]:
		investigations(take)