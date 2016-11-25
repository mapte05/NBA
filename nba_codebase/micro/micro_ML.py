# ________________________________________________________________________________
# Load Dictionaries with all Features
import pickle
with open('final_micro_dict.pickle', 'rb') as handle:
      master_team_micro_running_dict = pickle.load(handle)

with open('master_num_date_map.pickle', 'rb') as handle:
      master_num_date_map = pickle.load(handle)

with open('master_running_team_dict.pickle', 'rb') as handle:
      master_running_team_dict = pickle.load(handle)
# ________________________________________________________________________________

# Print out all Offensive and Defensive Stats Categories
for team, team_dict in master_team_micro_running_dict.items():

    for game_index, date in master_num_date_map[team].items():

        if game_index == 1:
            continue

        print team_dict[date]["OFF"].keys()
        print team_dict[date]["DEF"].keys()

        break

    break