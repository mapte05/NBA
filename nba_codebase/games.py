import csv
import sys
import numpy
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from datetime import date
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

filename = "game_logs_1415.csv"
# Raw game information per team
master_raw_team_dict = {}
# Map for each team of game number (indexed 1->82 for 82 game season) to the date of that game
master_num_date_map = {}

header = None

with open(filename, 'rb') as f:
    reader = csv.reader(f)

    team_tracker = set([])

    for ind,row in enumerate(reader):

    	# Grab table headers
    	if ind == 0:
    		header = row
    		continue

    	# Convert Row to Game Number
        game_number = 83-(((ind-1) % 82)+1)

        # _____________________________________________________________
        # Extract team name, date, H vs. A from Matchup
        matchup_form = row[0][row[0].find("-")+1:].split()
        matchup_date = row[0][:row[0].find("-")-1]
        date_split = row[0].replace(",","")[:row[0].find("-")-1].split()

        team_name = matchup_form[0]
        opp_team_name = matchup_form[2]
        home_away = "H"
        if matchup_form[1] == "@":
        	home_away = "A"
        # _____________________________________________________________

        # Initialize Raw dictionary. "Info" holds text information, "Stats" hold numeric information.
        if team_name not in team_tracker:
        	team_tracker.add(team_name)
        	master_raw_team_dict[team_name] = {}
        	master_num_date_map[team_name] = {}

        master_raw_team_dict[team_name][matchup_date] = {}
        master_raw_team_dict[team_name][matchup_date]["Info"] = {}

        game_stats = {col_name: row[col] for col,col_name in enumerate(header)}
        for k,v in game_stats.items():
        	if "Matchup" in k or "W/L" in k:
        		master_raw_team_dict[team_name][matchup_date]["Info"][k] = v
        		game_stats.pop(k,None)
        		continue
        	game_stats[k] = float(v)

        master_raw_team_dict[team_name][matchup_date]["Info"]["TEAM"] = team_name
        master_raw_team_dict[team_name][matchup_date]["Info"]["OPPTEAM"] = opp_team_name
        master_raw_team_dict[team_name][matchup_date]["Info"]["LOC"] = home_away
        master_raw_team_dict[team_name][matchup_date]["Info"]["DATE"] = matchup_date
        master_raw_team_dict[team_name][matchup_date]["Info"]["DATE_MON"] = months.index(date_split[0])+1
        master_raw_team_dict[team_name][matchup_date]["Info"]["DATE_DAY"] = int(date_split[1])
        master_raw_team_dict[team_name][matchup_date]["Info"]["DATE_YEAR"] = int(date_split[2])

        master_num_date_map[team_name][game_number] = matchup_date
        master_raw_team_dict[team_name][matchup_date]["Stats"] = game_stats

# Running dictionary maps team to date to stats where its the running average of stats
# UP UNTIL (before) the date that is the key. e.x. master_running["GSW"]["Apr. 12, 2015"]["Stats"]
# is the average of GSW's stats for all games in the season before their game on Apr. 12, 2015
master_running_team_dict = {team:{} for team in master_raw_team_dict.keys()}

# All metrics is the header names for the offensive and defensive running average metrics
stats_headers = [metric for metric in header if "Matchup" not in metric and "W/L" not in metric]
# The metrics of all the teams the current team has played (for exampe, average Offensive Rebounds
# against them) are all delineated with an "OPP-"
all_metrics = stats_headers + ["OPP-" + entry for entry in stats_headers]

for team in master_raw_team_dict.keys():

	team_raw_dict = master_raw_team_dict[team]
	team_num_date_map = master_num_date_map[team]

	for game_index in range(1, 83):

		curr_game_date = team_num_date_map[game_index]
		curr_opp_team = team_raw_dict[curr_game_date]["Info"]["OPPTEAM"]

		master_running_team_dict[team][curr_game_date] = {}
		master_running_team_dict[team][curr_game_date]["Info"] = team_raw_dict[curr_game_date]["Info"]

		master_running_team_dict[team][curr_game_date]["Info"]["DELTA"] = \
		team_raw_dict[curr_game_date]["Stats"]["PTS"] - \
		master_raw_team_dict[curr_opp_team][curr_game_date]["Stats"]["PTS"]

		# No running average information before the 1st game of the season
		if game_index == 1:
			master_running_team_dict[team][curr_game_date]["Stats"] = { metric:0 for metric in all_metrics}
			# If adding new featues, make sure to initialize here.
			master_running_team_dict[team][curr_game_date]["Stats"]["W%"] = 0
			master_running_team_dict[team][curr_game_date]["Stats"]["REST"] = 0
			#
			continue

		new_game_stats = {}
		previous_date = team_num_date_map[game_index-1]
		prev_opp_team = team_raw_dict[previous_date]["Info"]["OPPTEAM"]
		master_running_team_dict[team][curr_game_date]["Stats"] = {}

		for metric in all_metrics:

			old_avg_value = master_running_team_dict[team][previous_date]["Stats"][metric]
			old_avg_value_mult = old_avg_value*(game_index-2)

			new_game_value = 0.0
			# If its an opposing team metric, go get the new value to add from the other team's raw info
			if "OPP" in metric:
				real_metric = metric.replace("OPP-", "")
				new_game_value = master_raw_team_dict[prev_opp_team][previous_date]["Stats"][real_metric]
			# If current team metric, get if from the previous game of your own raw info
			else:
				new_game_value = team_raw_dict[previous_date]["Stats"][metric]

			new_total = new_game_value + old_avg_value_mult
			new_avg_value = float(new_total)/(game_index-1)

			master_running_team_dict[team][curr_game_date]["Stats"][metric] = new_avg_value

		# ________________________________________________________________________________
		# Win % Add-On
		old_avg_value = master_running_team_dict[team][previous_date]["Stats"]["W%"]
		old_avg_value_mult = old_avg_value*(game_index-2)

		new_game_value = 0.0
		if team_raw_dict[previous_date]["Info"]["W/L"] == "W":
			new_game_value = 1.0

		new_total = new_game_value + old_avg_value_mult
		new_avg_value = float(new_total)/(game_index-1)

		master_running_team_dict[team][curr_game_date]["Stats"]["W%"] = new_avg_value
		# ________________________________________________________________________________
		# Days Rest Add-On

		team_raw_dict[previous_date]

		prev_d = date(team_raw_dict[previous_date]["Info"]["DATE_YEAR"] \
			, team_raw_dict[previous_date]["Info"]["DATE_MON"] \
			, team_raw_dict[previous_date]["Info"]["DATE_DAY"])

		curr_d = date(team_raw_dict[curr_game_date]["Info"]["DATE_YEAR"] \
			, team_raw_dict[curr_game_date]["Info"]["DATE_MON"] \
			, team_raw_dict[curr_game_date]["Info"]["DATE_DAY"])

		days_rest = (curr_d - prev_d).days - 1

		master_running_team_dict[team][curr_game_date]["Stats"]["REST"] = days_rest
# ________________________________________________________________________________________

# # Code to test against true Game Logs to make sure working properly.


# _____________________________________________________________________________________________

# SAMPLE MACHINE LEARNING

# You should only need to use master_running_team_dict because I think I copied over any important
# information from raw into running before this section.

# As a reminder, the "Stats" for a given game are the running values as calculated BEFORE that game.
# The "Info" for a game is what actually happened in that game (who won and by how much)

# Value to separate the train from the test set

data_matrix = numpy.zeros((81  * len(master_running_team_dict),40 * 2))
data_labels = []


# GET TRAINING MATRIX
data_matrix_index = 0
for game_index in range(2, 83):
    for team1 in master_running_team_dict.keys():
        team1_running_dict = master_running_team_dict[team1]
        # Start with game 2 (because game 1 has no information yet)
        # and go until the threshold value for training.
        game_date = master_num_date_map[team1][game_index]
        team2 = team1_running_dict[game_date]["Info"]["OPPTEAM"]

        team1_features = master_running_team_dict[team1][game_date]["Stats"]
        team2_features = master_running_team_dict[team2][game_date]["Stats"]
        # W or L
        binary_result_team1 = master_running_team_dict[team1][game_date]["Info"]["W/L"]
        binary_result_team2 = master_running_team_dict[team2][game_date]["Info"]["W/L"]

        # Modify the training matrix
        data_matrix[data_matrix_index] = numpy.concatenate((team1_features.values(),team2_features.values()), axis=0)
        data_matrix_index += 1
        data_labels.append(binary_result_team1)

# TESTING
svm_accuracy = []
nb_accuracy  = []
rf_accuracy = []

numpy.random.shuffle(data_matrix)
for i in range(10, 70):
    threshold = i * len(master_running_team_dict)
    # numpy.random.shuffle(data_matrix)

    training_matrix = data_matrix[:threshold,:]
    training_labels = data_labels[:threshold]

    test_matrix = data_matrix[threshold:,:]
    test_labels = data_labels[threshold:]

    #Do all the machine learning
    #TRAINING
    clf = svm.SVC(kernel='rbf')
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier(n_estimators=10)

    clf.fit(training_matrix, training_labels)
    clf2.fit(training_matrix, training_labels)
    clf3.fit(training_matrix, training_labels)
    # TESTING
    svm_predicted_labels = clf.predict(test_matrix)
    nb_predicted_labels = clf2.predict(test_matrix)
    rf_predicted_labels = clf3.predict(test_matrix)

    cmsvm = confusion_matrix(test_labels, svm_predicted_labels, labels=["W","L"])
    cmnb = confusion_matrix(test_labels, nb_predicted_labels, labels=["W","L"])
    cmrf = confusion_matrix(test_labels, rf_predicted_labels, labels=["W","L"])

    svm_accuracy.append((cmsvm[0,0] + cmsvm[1,1])*1.0/numpy.sum(cmsvm))
    nb_accuracy.append((cmnb[0,0] + cmnb[1,1])*1.0/numpy.sum(cmnb))
    rf_accuracy.append((cmrf[0,0] + cmrf[1,1])*1.0/numpy.sum(cmrf))

# CREATE PLOTS FOR PROGRESS REPORT

# PLOT 1 - The Win percentages for each team over time
# for team in master_raw_team_dict.keys():
#     win_percentages = []
#     for game_index in range(2,83):
#         win_percentages.append(master_running_team_dict[team][master_num_date_map[team][game_index]]["Stats"]["W%"])
#
#     plt.plot(range(2, 83),win_percentages, label=team)
#     plt.legend()
# plt.show()
# PLOT 2 - Accuracies for each classifier using different threshold values for num games

# plt.plot(range(10, 70),svm_accuracy, label="svm")
# plt.legend()
# plt.plot(range(10, 70),nb_accuracy, label="naive bayes")
# plt.legend()
# plt.plot(range(10, 70),rf_accuracy, label="random forest")
# plt.legend()
# plt.ylabel('classification accuracy')
# plt.xlabel('threshold values')
# plt.show()

###### THE MAX Accuracies for each classifier
print 'The Max accuracies are:', 'SVM: ', max(svm_accuracy), ' Naive Bayes:', max(nb_accuracy), ' Random Forest: ', max(rf_accuracy)


### PLOT 3  - Accuracices for random classifiers
lower_bound = 10 * len(master_running_team_dict)
upper_bound = 70 * len(master_running_team_dict)
plt.plot(xrange(lower_bound, upper_bound, len(master_running_team_dict)),svm_accuracy, label="svm")
plt.legend()
plt.plot(xrange(lower_bound , upper_bound, len(master_running_team_dict)),nb_accuracy, label="naive bayes")
plt.legend()
plt.plot(xrange(lower_bound, upper_bound, len(master_running_team_dict)),rf_accuracy, label="random forest")
plt.legend()
plt.ylabel('classification accuracy')
plt.xlabel('# of random games sampled')
plt.show()
