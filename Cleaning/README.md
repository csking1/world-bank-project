# Steps for Cleaning and Entity Resolution

1. reading.py: Read in contracts, projects, and currency conversion from World Bank Api and generate csvs.

2. entity.py and currency.py: Resolve entity conflicts and currency discrepancies.

3. summary.py: Generate summary statistics for projects, contracts, and investigations.

4. transform.py: Impute missing values and create clean csvs for projects, contracts, and investigations to hand to Features.

Note: The public files are too large to store on github, so they'll need to be gotten from the api for viewing.
