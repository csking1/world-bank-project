# Here's a file you can use to get the cleaned data running on your repository

1. ssh into the server

2. In the same directory, which is your home directory, clone our repo with
	```
	git clone https://github.com/csking1/world-bank-project.git
	```

3. Copy these two files into your Example directory. This has a .gitignore; it's important that niether of these csvs is put in a different folder, because they can't exit the server. Note that you'll have to change USERNAME to your own.
	```
	cp /mnt/data3/world-bank/pipeline_data/complaint_case_projects_and_contracts_20150727_115323.csv /home/USERNAME/world-bank-project/Example/investigations_one.csv
	cp /mnt/data3/world-bank/pipeline_data/utf8_new_complaint_data_20150727.csv /home/USERNAME/world-bank-project/Example/investigations_two.csv
	```

4. Make sure you have two csvs in Example, investigations_one.csv and investigations_two.csv. You should also have a .gitignore.

5. Cd into the Cleaning directory and run the following command
	```
	./etl.sh
	```
	This file runs all of my cleaning scripts, and should give you a cleaned labeled csv in the Example folder. Let me know if anything breaks. After copying the files - that will always stay on the server - you can run the package with just this last command.
