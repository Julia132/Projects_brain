"""""
This module activates the program

############################

The program is comprised of three basic modules:

brain_main_git - multiple stage processing and analysis of inner contours

brain_dataset_git - counting various geometric features in the contours

csv_git - classification geometric features from .csv, analysis of job
"""""

from brain_main_git import main
main()

from brain_dataset_git import dataset
dataset()

from csv_git import csv_model
csv_model()
