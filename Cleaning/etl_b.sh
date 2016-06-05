#!/bin/bash

echo Cleaning Data Sets

python cleaning.py Data/contracts.csv Output/contracts.csv
python cleaning.py Data/projects.csv Output/projects.csv
python cleaning.py ../Example/investigations_one.csv Output/investigations_one.csv
python cleaning.py ../Example/investigations_two.csv Output/investigations_two.csv

echo Resolving Entities
python -W ignore "entity_resolution.py" -c 'Output/contracts.csv' -e 'Data/names.csv' -o 'Output/contracts_resolved.csv'

echo Joining DataFrames in Pandas
python join_b.py Output/contracts_resolved.csv Output/projects.csv Output/investigations_one.csv Output/resolved_joined_b.csv

echo Cleaning the resolved and joined file
python cleaning.py Output/resolved_joined_b.csv ../Example/resolved_joined_b.csv
echo The file Example/resolved_joined.csv is ready for summary and feature generation