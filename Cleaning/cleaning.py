import csv
import pandas as pd
import sys

def clean(df, outfile):
	'''
	Takes a pandas data frame, replaces null values, and writes out a cleaned version to the csv
	'''
	new = df.fillna(df.mode().iloc[0])
	if "project" in outfile:
		for each in ['projectdoc ', "majorsector_percent ", "mjsector1", "financier", "mjsector2", "mjsector3", "mjsector4", "mjsector5", "theme ", "mjtheme1name", "mjtheme2name", "mjtheme3name", "mjtheme4name", "mjtheme5name", "Unnamed: 56"]:
			del new[each]
	elif "investigations_one" in outfile:
		for each in ["Unnamed: 35", "Unnamed: 37", "Unnamed: 38"]:
			del new[each]

	new.to_csv(outfile)

if __name__ == "__main__":
	df = pd.read_csv(sys.argv[1])
	clean(df, sys.argv[2])