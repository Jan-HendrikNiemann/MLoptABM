#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:01:02 2022

Script to adjust the python files to a given virtual environment

@author: Jan-Hendrik Niemann
"""


import os


# Shebang to adjust
new_shebang = '#!<path_to_GERDA_virtuel_environment>/envs/gerdaenv/bin/python3.8'
old_shebang = '#!<path_to_GERDA_virtuel_environment>/envs/gerdaenv/bin/python3.8'

# Path to opt_settings.txt to adjust
new_path = '<path_to_input_and_output_directory>/opt_settings.txt'
old_path = '<path_to_input_and_output_directory>/opt_settings.txt'

# List to store files
file_list = []

# Iterate directory
for file in os.listdir(os.getcwd()):
    if file.endswith('.py') and not file.startswith('.'):
        file_list.append(file)

# Adjust files
for filename in file_list:
    if filename == os.path.basename(__file__):
        continue

    # Read file
    with open(filename, 'r') as file:
        content_old = file.read()

    # Replace strings
    content_new = content_old.replace(old_shebang, new_shebang)
    content_new = content_new.replace(old_path, new_path)

    if content_old == content_new:
        print('%s --> nothing to change' % filename)
    else:
        print('%s --> adjusted' % filename)
        # Write new content
        with open(filename, 'w') as file:
            file.write(content_new)
