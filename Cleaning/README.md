# Steps for Cleaning and Entity Resolution

1. reading.py: Read in contracts, projects, and currency conversion from World Bank Api and generate csvs. Does this need to grab the most recent files for the pipeline, or should we just read from the csvs we already have?

2. summary.py: Generate summary statistics for projects, contracts, and investigations.
	- To run the file, pass in csvs from the Data/ directory for public information or Example/ for proprietary, which is only available on our private server.

	```
	python summary.py Data/contracts.csv
	```

3. transform.py: Impute missing values and create clean csvs for projects, contracts, and investigations to hand to Features.

4. entity.py and currency.py: Resolve entity conflicts and currency discrepancies.
	```
	python -W ignore "entity_resolution.py" -c 'Data/contracts.csv' -e 'Data/names.csv' -o '../Example/contracts.csv'
	```

Note: The public files are too large to store on github, so they'll need to be gotten from the api for viewing.
