#!/usr/bin/python

# This script will iterate through all of the data files in the arguments
# and paste a randomized collection of these tweets into a file for
# training.
# Use the -i flag to indicate the input file. It is assumed
#    that this directory contains a list of CSV files.
# Use the -o flag to indicate the output directory.

import os
import sys, getopt

def main(argv):
    input_file = ''
    output_file = ''
   
    # This try/except block gets the relevant input/output files
    # to references for breaking up the files.
    try:
       opts, args = getopt.getopt(argv, "hi:o:", ["ifile=","ofile="])
    except getopt.GetoptError:
        print("invalid input")
    for opt, arg in opts:
        if opt == '-h':
            print('python3 select_random.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg

    # First check to make sure that there is an input file 
    # and an output file
    if(input_file == ''):
        print("A valid directory must be provided using the '-i' flag.")
        sys.exit()
    if(output_file == ''):
        print("A valid file name must be provided for generating the output file using the '-o' flag.")
        sys.exit()     

    
    # Assume the file path is relative.
    
    files = os.listdir(input_file)

    output_lines = []
    for f in files:

        with open('/'.join([input_file, f]), 'r') as fp:
            for i, line in enumerate(fp):
                if i % 5 == 0:
                    output_lines.append(line)

    with open(output_file, 'w') as new_file:  
        new_file.writelines(output_lines)

if  __name__  == "__main__":
    main(sys.argv[1:])




