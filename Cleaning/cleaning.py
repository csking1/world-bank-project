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
			try:
				del new[each]
			except:
				pass
		new.rename(columns = {'id': s}, inplace = True)
	elif "investigations_one" in outfile:
		for each in ["Unnamed: 35", "Unnamed: 37", "Unnamed: 38"]:
			try:
				del new[each]
			except:
				pass
		new = new.drop_duplicates(["Project Number"], take_last = True )
		new.rename(columns = {'Project Number': s}, inplace = True)
	elif "investigations_two" in outfile:
		new = new.drop_duplicates(["project_number"], take_last = True)
		new.rename(columns = {'project_number': s}, inplace = True)
	elif "resolved" in outfile:
		for each in ["Borrower", "Contract Number",	"Case Number",	"CaseOutcome",	"Closing Date",	"Complaint No",	"Complaint Status",	"Concept Review Date",	"Contract Amount",	"Contract Name",	"Contract Sign-off Date",	"Country_x", "Date Case Opened",	"Date Complaint Opened",	"Decision Meeting Date",	"Effectiveness Date",	"Lead Investigator", "Major Sector_y", "No Objection Date", "Num days from col 16 to col 1",	"Procurement Method ID",	"Procurement Type Description",	"Project Amount",	"Project", "Name_y",	"Signing Date", "SubjectName", "Supplier_y", "Unnamed: 0_x", "Unnamed: 0.1_y", "VPU","WB ID",	"envassesmentcategorycode",	"supplementprojectflg", "productlinetype", "url", "Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1_x" , "WB Contract Number", "Project ID", "Borrower Contract Reference Number", "orig_supplier_name", "supplier_x", "complaint_no", "wb_id", "Unnamed: 0_y", "countryname", "project_name_x", "subjectname", "status", "project_name_y", "location", "GeoLocID", "GeoLocName", "Latitude", "Longitude" , "Country_y", "supplier_y", "Project Name_x", "As of Date", "Supplier_x", "Contract Description", "borrower_contract_number", "case_number", "contract_name", "projectstatusdisplay", "sector1", "sector2", "sector3", "sector4","sector5","sector", "sector", "mjsector", "theme1", "theme2",	"theme3",	"theme4", "theme5", "goal", 'Allegation OutCome', "Allegation Category", "Allegation Type", "Approval Date", "Bank Approval Date", "Bank Appraisal Date ", "Begin Preparation Date"]:
			try:
				del new[each]
			except:
				pass

	print ("Sending csv to file")
	new.to_csv(outfile)

if __name__ == "__main__":
	df = pd.read_csv(sys.argv[1])
	clean(df, sys.argv[2])