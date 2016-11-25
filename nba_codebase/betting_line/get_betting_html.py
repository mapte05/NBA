# scrapes covers.com to get point spread data
# usage: in terminal - "python get_betting_html.py"
# desired output: outputs a "betting_line_html.pickle" file that stores the html for each day there was
# 				  a basketball game during the 2014-15 season



import csv
import sys
from datetime import date
import pickle
import requests  
import json
import collections

# GLOBAL CONSTANTS --------------------------------------

MONTHS ={
			"JAN": "1",
			"FEB": "2", 
			"MAR": "3",
			"APR": "4",
			"MAY": "5",
			"JUN": "6",
			"JUL": "7",
			"AUG": "8",
			"SEP": "9",
			"OCT": "10",
			"NOV": "11",
			"DEC": "12"
		}

# -------------------------------------------------------------


def requestBettingData(date):
	# parse NOV 08, 2014 format into 2014-08-11
	splitDate = date.split()
	month = MONTHS[splitDate[0]]
	day = splitDate[1][:-1] # slice off comma
	year = splitDate[2]
	urlDate = year + "-" + day + "-" + month

	# make request to html
	try:
		url = "http://www.covers.com/sports/NBA/matchups?selectedDate=" + urlDate
		r = requests.get(url)
		r.raise_for_status()
		return r.text
	except requests.exceptions.HTTPError:
		print "status code: %i" % r.status_code
		print "invalid url: %s" % url


def getAllDatesFromPickle():
	result = set()
	with open('../micro/master_num_date_map.pickle', 'rb') as handle: #todo: make this file read less sketchy
		master_num_date_map = pickle.load(handle)
	for team in master_num_date_map:
		for game, date in master_num_date_map[team].items():
			result.add(date)
	return result



def main():
	print ""
	print "if you really need scrape again, then comment out the code in main and rerun"
	print "hopefully the html is already in betting_line_html.pickle"
	print ""
	
	# print "starting to scrape http://www.covers.com/sports/NBA/matchups?selectedDate=[DATE]"
	# print ""
	# result = []
	# dates = getAllDatesFromPickle()
	# data = {} # date -> html dump for all matchups on that day
	# for date in dates:
	# 	data[date] = requestBettingData(date)
	# 	print "finished " + date
	# with open('betting_line_html.pickle', 'wb') as handle:
	# 	pickle.dump(data, handle)

	# print ""
	# print "we scraped that puppy well done"



if __name__ == '__main__':
    main()