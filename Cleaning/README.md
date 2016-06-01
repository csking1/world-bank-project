# Steps for Cleaning and Entity Resolution

1. Download public data from the World Bank API using the commands in reading.py

2. Cleaning.py: This is called once for each of the contracts, projects, and investigations csvs. It writes the cleaned data to a csv in Output
	- To run the file, pass in csvs from the Data/ directory

	```
	python cleaning.py Data/contracts.csv Output/contracts.csv
	```

3. Resolve the discrepancies with entity resolution using this command and file
	```
	python -W ignore "entity_resolution.py" -c 'Data/contracts.csv' -e 'Data/names.csv' -o '../Example/contracts.csv'
	```

4. Read in the cleaned csvs, join them using pandas, and write out one csv to send to feature generation
	```
	python join.py Output/contracts.csv Output/projects.csv Output/investigations.csv ../Example/landing.csv
	```

5. Generate summary statistics on the joined csv using
	```
	python summary.py ../Example/landing.csv Output/summary.txt
	```