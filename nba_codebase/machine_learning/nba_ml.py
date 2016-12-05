import pickle
import numpy
# import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier

class NBAMachineLearning:

    def __init__(self,micro_dict, date_map, team_dict, betting_lines):
        self.master_team_micro_running_dict = micro_dict
        self.master_num_date_map = date_map
        self.master_running_team_dict = team_dict
        self.master_betting_line_dict = betting_lines
        self.init_data()
        self.build_data()

    def init_data(self):
        num_games = (83-2) * len(master_running_team_dict) - 70
        num_features = 416
        self.data_matrix = numpy.zeros((num_games,num_features))
        self.data_labels = []
        self.data_betting_lines = []
        self.data_matrix_index = 0
        self.inaddmissable_games = 0


    def build_data(self):
        for game_index in range(2, 83):
            teams_seen = set()
            for team1 in master_running_team_dict.keys():
                if team1 not in teams_seen:
                    team1_running_dict = master_running_team_dict[team1]
                    # and go until the threshold value for training.
                    game_date = master_num_date_map[team1][game_index]
                    team2 = team1_running_dict[game_date]["Info"]["OPPTEAM"]
                    teams_seen.add(team2)
                    # gather all features
                    team1_features = master_running_team_dict[team1][game_date]["Stats"]
                    team2_features = master_running_team_dict[team2][game_date]["Stats"]
                    team1_offensive_micro_features = master_team_micro_running_dict[team1][game_date]["OFF"]
                    team1_defensive_micro_features = master_team_micro_running_dict[team1][game_date]["DEF"]
                    team2_offensive_micro_features = master_team_micro_running_dict[team2][game_date]["OFF"]
                    team2_defensive_micro_features = master_team_micro_running_dict[team2][game_date]["DEF"]
                    # combined the features
                    team1_micro_concatenated = numpy.concatenate((team1_offensive_micro_features.values(),team1_defensive_micro_features.values()))
                    team2_micro_concatenated = numpy.concatenate((team2_offensive_micro_features.values(), team2_defensive_micro_features.values()))
                    micro_concatenated = numpy.concatenate((team1_micro_concatenated, team2_micro_concatenated))
                    stats_concatenated = numpy.concatenate((team1_features.values(), team2_features.values()))
                    final_features = numpy.concatenate((stats_concatenated, micro_concatenated))
                    game_result = master_running_team_dict[team1][game_date]["Info"]["W/L"]
                    betting_line = master_betting_line_dict[team1][game_date]
                    if final_features.shape[0] == 416:
                        self.data_matrix[self.data_matrix_index] = final_features
                        self.data_labels.append(game_result)
                        self.data_betting_lines.append(betting_line)
                        self.data_matrix_index += 1
                    else:
                        self.inaddmissable_games += 1
        self.data_matrix = numpy.nan_to_num(self.data_matrix)

    def run_classification(self):
        # MACHINE LEARNING
        svm_accuracy = []
        nb_accuracy  = []
        rf_accuracy = []
        perceptron_accuracy = []

        # Varying threshold values
        for i in range(60,61):
            threshold = i * len(master_running_team_dict)
            # Divide up the data
            training_matrix = self.data_matrix[:threshold,:]
            training_labels = self.data_labels[:threshold]
            test_matrix = self.data_matrix[threshold:,:]
            test_labels = self.data_labels[threshold:]

            # TRAINING
            clf = svm.SVC(kernel='rbf')
            clf2 = GaussianNB()
            clf3 = RandomForestClassifier(n_estimators=10)
            clf4 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)

            clf.fit(training_matrix, training_labels)
            clf2.fit(training_matrix, training_labels)
            clf3.fit(training_matrix, training_labels)
            clf4.fit(training_matrix, training_labels)

            # Predictions
            svm_predicted_labels = clf.predict(test_matrix)
            nb_predicted_labels = clf2.predict(test_matrix)
            rf_predicted_labels = clf3.predict(test_matrix)
            perceptron_predicted_labels = clf4.predict(test_matrix)

            cmsvm = confusion_matrix(test_labels, svm_predicted_labels, labels=["W","L"])
            cmnb = confusion_matrix(test_labels, nb_predicted_labels, labels=["W","L"])
            cmrf = confusion_matrix(test_labels, rf_predicted_labels, labels=["W","L"])
            cmpercep = confusion_matrix(test_labels, perceptron_predicted_labels, labels=["W","L"])

            svm_accuracy.append((cmsvm[0,0] + cmsvm[1,1])*1.0/numpy.sum(cmsvm))
            nb_accuracy.append((cmnb[0,0] + cmnb[1,1])*1.0/numpy.sum(cmnb))
            rf_accuracy.append((cmrf[0,0] + cmrf[1,1])*1.0/numpy.sum(cmrf))
            perceptron_accuracy.append((cmpercep[0,0] + cmpercep[1,1])*1.0/numpy.sum(cmpercep))
        print max(svm_accuracy), max(nb_accuracy), max(rf_accuracy), max(perceptron_accuracy)

    def 




if __name__ == "__main__":
# File Processing
    with open('../micro/final_micro_dict.pickle', 'rb') as pickle_file:
        master_team_micro_running_dict = pickle.load(pickle_file)
    with open('../micro/master_num_date_map.pickle', 'rb') as pickle_file:
        master_num_date_map = pickle.load(pickle_file)
    with open('../micro/master_running_team_dict.pickle', 'rb') as pickle_file:
        master_running_team_dict = pickle.load(pickle_file)
    with open('../betting_line/betting_lines.pickle', 'rb') as pickle_file:
        betting_line_dict = pickle.load(pickle_file)
    nba_class = NBAMachineLearning(master_team_micro_running_dict,master_num_date_map,master_running_team_dict,betting_line_dict)
    nba_class.run_classification()

# Build the training matrix




# Feature Selection
# Model Training
# Validation
