import csv
import sys
from datetime import date
import pickle
import requests  
import json
import collections


with open('master_num_date_map.pickle', 'rb') as handle:
      master_num_date_map = pickle.load(handle)

with open('master_running_team_dict.pickle', 'rb') as handle:
      master_running_team_dict = pickle.load(handle)

# __________________________________________________________________________________________

team_filename = "nba_team_id.csv"

abv_to_name_id = {}
name_to_abv_id = {}

with open(team_filename, 'rb') as f:
    reader = csv.reader(f)

    for ind,row in enumerate(reader):

        abv_to_name_id[row[0]] = (row[1],row[2])
        name_to_abv_id[row[1]] = (row[0],row[2])

# __________________________________________________________________________________________

def find_stats(start_date, end_date, team, team_id, running):

    start_date_list = start_date.replace(",","").split()
    end_date_list = end_date.replace(",","").split()

    print team, start_date_list, end_date_list
    
    start_date_url = start_date_list[0] + "%2F" + start_date_list[1] + "%2F" + start_date_list[2]
    end_date_url = end_date_list[0] + "%2F" + end_date_list[1] + "%2F" + end_date_list[2]

    url = 'http://stats.nba.com/stats/teamdashptshots?' + \
    'DateFrom='+ start_date_url +'&DateTo=' + end_date_url + \
    '&GameSegment=&LastNGames=0&LeagueID=00&Location=&' + \
    'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' + \
    'PaceAdjust=N&PerMode=PerGame&Period=0&PlusMinus=N&Rank=N&' + \
    'Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&' + \
    'TeamID=' + str(team_id) + '&VsConference=&VsDivision='


    u_a = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 Safari/537.36"
    response = requests.get(url, headers={"USER-AGENT":u_a})
    response.raise_for_status() # raise exception if invalid response
    # shots = response.json()['resultSets'][0]['rowSet']
    data = json.loads(response.text)

    if running:
        team_date_running_dict[team][end_date] = data
    else:
        team_date_download_dict[team][end_date] = data
        

master_team_date_download = {}
master_team_date_running = {}

if len(sys.argv) == 1:

    with open('game.pickle', 'rb') as handle:
      master_team_date_download = pickle.load(handle)

    with open('running.pickle', 'rb') as handle:
      master_team_date_running = pickle.load(handle)

    pass

else:

    team_date_download_dict = {abv: {} for abv in abv_to_name_id.keys()}
    team_date_running_dict = {abv: {} for abv in abv_to_name_id.keys()}

    for team, num_date_map in master_num_date_map.items():

        start_date = num_date_map[1]
        for index, end_date in num_date_map.items():
            team_id = abv_to_name_id[team][1]

            if sys.argv[1] == "1":
                find_stats(end_date, end_date, team, team_id, False)
            elif sys.argv[1] == "2":
                find_stats(start_date, end_date, team, team_id, True)

    if sys.argv[1] == "1":
        with open('game.pickle', 'wb') as handle:
            pickle.dump(team_date_download_dict, handle)

    elif sys.argv[1] == "2":
        with open('running.pickle', 'wb') as handle:
            pickle.dump(team_date_running_dict, handle)

# ____________________________________________________________________________________________

master_team_micro_running_dict = {team:{} for team in master_team_date_download.keys()}

for team in master_team_date_download.keys():

    team_download = master_team_date_download[team]
    team_running = master_team_date_running[team]
    team_date_map = master_num_date_map[team]

    for game_index in range(1,83):

        date = team_date_map[game_index]

        print team, game_index, date

        master_team_micro_running_dict[team][date] = {}
        master_team_micro_running_dict[team][date]["OFF"] = collections.defaultdict(float)
        master_team_micro_running_dict[team][date]["DEF"] = collections.defaultdict(float)

        if game_index == 1:
            continue

        previous_date = team_date_map[game_index-1]

        offense_dict = {}
        for section_ind in range(6):

            if section_ind == 4 or section_ind == 2 or section_ind == 5: 
                continue

            col_names = master_team_date_download[team][previous_date]['resultSets'][section_ind]['headers']
            section_name = col_names[4]
            col_names = col_names[9:]
            del col_names[7]
            del col_names[3]

            my_rows_len = len(master_team_date_download[team][previous_date]['resultSets'][section_ind]['rowSet'])
            for row_ind in range(my_rows_len):

                run_row = master_team_date_running[team][previous_date]['resultSets'][section_ind]['rowSet'][row_ind]
                row_name = run_row[4]

                if row_name in ["ShotClock Off", "18-15 Early", "7-4 Late", "Not Captured"]:
                    continue

                run_row = run_row[9:]
                del run_row[7]
                del run_row[3]

                off_entry_name = row_name

                for col_ind in range(len(col_names)):
                    key = off_entry_name + " (" + col_names[col_ind] + ")"
                    offense_dict[key] = run_row[col_ind]

            opp_team = master_running_team_dict[team][previous_date]["Info"]["OPPTEAM"]
            opp_rows_len = len(master_team_date_download[opp_team][previous_date]['resultSets'][section_ind]['rowSet'])
            for row_ind in range(opp_rows_len):

                opp_down_row = master_team_date_download[opp_team][previous_date]['resultSets'][section_ind]['rowSet'][row_ind]
                row_name = opp_down_row[4]

                if row_name in ["ShotClock Off", "18-15 Early", "7-4 Late", "Not Captured"]:
                    continue

                opp_down_row = opp_down_row[9:]
                del opp_down_row[7]
                del opp_down_row[3]

                off_entry_name = row_name
                if section_ind == 4:
                    off_entry_name = "10+ Shots " + row_name
                def_entry_name = "DEF-" + off_entry_name

                opp_down_stats_dic = {def_entry_name + " (" + col_names[col_ind] + ")" :opp_down_row[col_ind] for col_ind in range(len(col_names))}

                for metric, metric_value in opp_down_stats_dic.items():

                    old_avg_value = master_team_micro_running_dict[team][previous_date]["DEF"][metric]

                    old_avg_value_mult = old_avg_value*(game_index-2)

                    new_avg_value = 0.0
                    if metric_value == None:
                        new_avg_value = old_avg_value
                    else:
                        new_total = metric_value + old_avg_value_mult
                        new_avg_value = float(new_total)/(game_index-1)

                    master_team_micro_running_dict[team][date]["DEF"][metric] = new_avg_value

        master_team_micro_running_dict[team][date]["OFF"] = offense_dict

# with open('final_micro_dict.pickle', 'wb') as handle:
#     pickle.dump(master_team_micro_running_dict, handle)
