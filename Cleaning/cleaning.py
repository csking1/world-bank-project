import csv
import pandas as pd
import sys

def clean(df, outfile):
	'''
	Takes a pandas data frame, replaces null values, and writes out a cleaned version to the csv
	'''
	s = "Project ID"
	new = df.fillna(df.mode().iloc[0])
	if "project" in outfile:
		for each in ['projectdoc ', "majorsector_percent ", "mjsector1", "financier", "mjsector2", "mjsector3", "mjsector4", "mjsector5", "theme ", "mjtheme1name", "mjtheme2name", "mjtheme3name", "mjtheme4name", "mjtheme5name", "Unnamed: 56"]:
			del new[each]
		new.rename(columns = {'id': s}, inplace = True)
	elif "investigations_one" in outfile:
		for each in ["Unnamed: 35", "Unnamed: 37", "Unnamed: 38"]:
			del new[each]
		new = new.drop_duplicates(["Project Number"], take_last = True )
		new.rename(columns = {'Project Number': s}, inplace = True)

	elif "investigations_two" in outfile:
		new = new.drop_duplicates(["project_number"], take_last = True )
		new.rename(columns = {'project_number': s}, inplace = True)

	elif "resolved" in outfile:
		for each in ['Allegation OutCome', "Allegation Category", "Allegation Type", "Approval Date", "Bank Approval Date", "Bank Appraisal Date ", "Begin Preparation Date"]:
			try:
				del new[each]
			except:
				pass

	print ("Sending csv to file")
	new.to_csv(outfile)

if __name__ == "__main__":
	df = pd.read_csv(sys.argv[1])
	clean(df, sys.argv[2])