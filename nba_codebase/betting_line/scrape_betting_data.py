# scrapes covers.com to get point spread data
# usage: in terminal - "python scrape_betting_data.py"
# desired result: outputs a dump of team -> date -> opponent -> point spread 
# 			betting line data from the 2014-15 NBA season, in file "betting_lines.pickle"

import csv
import sys
from datetime import date
import pickle
import requests  
import json
import collections
from lxml import html
import re


ODD_NOT_AVAILABLE = "N/A"


def getTeamNames(tree):
	teamNames = tree.xpath('//div[@class="cmg_team_name"]/text()')
	teamNamesProcessed = []
	for name in teamNames:
		name = name.strip()
		if name != "":
			teamNamesProcessed.append(name)
	result = []
	for i in xrange(0, len(teamNamesProcessed), 2):
		matchup = (teamNamesProcessed[i], teamNamesProcessed[i+1])
		result.append(matchup)
	return result


def getOdds(htmlData):
	matches = re.findall("data-game-odd=\"(.*?)\"", htmlData)
	odds = []
	for m in matches:
		if m == '':
			odds.append(ODD_NOT_AVAILABLE)
		else:
			odds.append(m)
	return odds


def main():
	# get dict with date -> html
	with open('betting_line_html.pickle', 'rb') as handle:
		dates_to_html = pickle.load(handle)
	
	# process html to create team -> date -> opponent -> point spread from team's perspective
	master_point_spread = {}
	for date in dates_to_html:
		print date
		htmlData = dates_to_html[date].encode('utf-8')
		tree = html.fromstring(htmlData)
		matchups = getTeamNames(tree) # get the team names: [team, opp-team, team, opp-team, ...]
		odds = getOdds(htmlData) # get betting line odds via regex, N.B. some odds weren't available
		assert len(matchups) == len(odds) # sanity check, assume that they follow the same order

		# cycle through both and add to to dict
		for i, teams in enumerate(matchups):
			pointSpread = odds[i]
			# first is away team, second is home team
			# a positive point spread means home team not favored to win, negative means is favored to win
			if teams[1] not in master_point_spread:
				master_point_spread[teams[1]] = {}
			if date not in master_point_spread[teams[1]]:
				master_point_spread[teams[1]][date] = {}
			master_point_spread[teams[1]][date][teams[0]] = pointSpread
			
			negatedPointSpread = ""
			if pointSpread.find("-") != -1: # means the spread for home team is negative
				negatedPointSpread = pointSpread[1:]
			else:
				negatedPointSpread = "-" + pointSpread
			
			if teams[0] not in master_point_spread:
				master_point_spread[teams[0]] = {}
			if date not in master_point_spread[teams[0]]:
				master_point_spread[teams[0]][date] = {}
			master_point_spread[teams[0]][date][teams[1]] = negatedPointSpread

	# dump dict with point spread odds into pickle file
	with open('betting_lines.pickle', 'wb') as handle:
		pickle.dump(master_point_spread, handle)



if __name__ == '__main__':
    # main()
    with open('betting_lines.pickle', 'rb') as handle:
    	odds_dict = pickle.load(handle)
	
	for team in odds_dict:
		print team
		print odds_dict[team]
		print ""



"""
notes:
		matchups = tree.xpath('//div[@class="cmg_matchup_header_team_names"]/text()')
		matchups = [entry.strip() for entry in matchups]
		print matchups


"""

