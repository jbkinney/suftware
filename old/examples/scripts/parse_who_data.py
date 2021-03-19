#!/usr/bin/env python3
import pandas as pd
import glob
import os

print('Running parse_who_data.py ...')

# Specify directories
raw_data_file = '../raw_data/who_data_2017.xlsx'
output_dir = '../data'

# Remove all previous WHO output files
files_to_remove = glob.glob('%s/who.*.txt'%output_dir)
for file in files_to_remove:
    os.remove(file)

# Load raw data
df = pd.read_excel(raw_data_file, header=None, skiprows=4)
df.head(10)

descriptions = df.iloc[0, 1:].values
units = df.iloc[1, 1:].values
names = df.iloc[2, 1:].values
years = df.iloc[3, 1:].values
num_datasets = df.shape[1] - 1

for i in range(num_datasets):
    # Extract data values
    values = df.iloc[5:, i + 1].values.astype(float)
    description_str = repr(descriptions[i] + ' (%s)' % years[i])

    # Format file contentx
    header = '# "description": %s\n# "name": "%s"\n# "units": "%s"\n' % \
             (description_str, names[i], units[i])
    contents = ''.join(['%.1f\n' % x for x in values])

    # Write file
    file_name = '%s/who.%s.txt' % (output_dir, names[i])
    print('Writing %s' % file_name)
    with open(file_name, 'w') as f:
        f.write(header)
        f.write(contents)

print('Done.')