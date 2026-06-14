#Use wc_all_matches to train the model and maybe use wc_top_scorers as credibility or somthing. 
#You could probably use the wc_all_editions to test the model or trainning. 
#You could use the wc_2026_matches to predict the winners of the 2026 world cup.
#its in strings so be careful 

#I Just listing my thought here, You all can ignore this...
#I relized that we can have two separed model, One for match, one for the overall tornment. We can input all the data for winners and like other places for each team as the x for each year, Then we use the winner of the tornment as the y. 
#We can use each match for the match predicotr, Input team 1 and team2 and fifa rank for both, and Idk if we shoudl do all the games... Porabblyy eevrything after 2000s or something. But use the winner thing as the y.

import json
from flask import Flask, request, render_template
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier 
from sklearn.datasets import load_iris

match_x = []
match_y = []
match_model = HistGradientBoostingClassifier()

# METHODS
def get_fifa_rank():
    data = []
    with open('wc_2026_teams.json', "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

#This creates another key called winner which determines which of the two teams won the match. 0 being the team1 won and 1 being team2 won and 3 being a draw. 
def match_winner() :
    with open('wc_all_matches.json', "r", encoding="utf-8") as f:
        data = json.load(f)
    # Python
    for item_dict in data:
        if item_dict["score1"] > item_dict["score2"]:
            item_dict["winner"] = 0
        elif item_dict["score1"] < item_dict["score2"]:
            item_dict["winner"] = 1
        else:
            item_dict["winner"] = 3  
    

    return data
def train_model(all_matches):
    global match_x, match_y, match_model
    match_x = pd.DataFrame({
        "team1": [item_dict["team1"] for item_dict in all_matches],
        "team2": [item_dict["team2"] for item_dict in all_matches],
    })
    #match_y = pd.DataFrame(match_y)
    #X_train, X_test, y_train, y_test = train_test_split(match_x, match_y, test_size=0.2, random_state=42)
    #match_model.fit(X_train, y_train)
    #y_pred = match_model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy:", accuracy)
#FLASK APP
app = Flask(__name__)

@app.route('/')
def home():
    #Train model here?
    return render_template('index.html')

@app.route('/match_predictor', methods=['POST','GET'])
def match_predictor():
    global match_x, match_y, match_model

    
    a = get_fifa_rank()
    desired_keys = ["fifa_rank", "team"]
    extracted_data = [{key: item[key] for key in desired_keys} for item in a]
    match_x = pd.DataFrame(extracted_data)
    
    b = match_winner()
    return b

@app.route('/wc_predictor', methods=["POST","GET"])
def wc_predictor():
    return "Hello World 2"

if __name__ == '__main__':
    app.run(debug=True)



