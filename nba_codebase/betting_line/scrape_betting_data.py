# scrapes covers.com to get point spread data
# usage: in terminal - "python scrape_betting_data.py"
# desired result: outputs a dump of team matchups -> point spread betting line data from the 2014-15 NBA season

import csv
import sys
from datetime import date
import pickle
import requests  
import json
import collections


def main():
	with open('betting_line_html.pickle', 'rb') as handle:
		dates_to_html = pickle.load(handle)
	print dates_to_html


if __name__ == '__main__':
    main()



"""
notes:

current goal: pull in all html and save as files, associating date with the html
format for date in dict is: FEB 20, 2015
The regular season began on Tuesday, October 28, 2014
The regular season ended on Wednesday April 15, 2015 

use html.parse to get from file
lets test this
"""

