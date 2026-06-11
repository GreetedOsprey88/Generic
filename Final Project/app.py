#Use wc_all_matches to train the model and maybe use wc_top_scorers as credibility or somthing. 
#You could probably use the wc_all_editions to test the model or trainning. 
#You could use the wc_2026_matches to predict the winners of the 2026 world cup.
#its in strings so be careful 

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
    
    print(match_x)
    return extracted_data

@app.route('/wc_predictor', methods=["POST","GET"])
def wc_predictor():
    return "Hello World 2"

if __name__ == '__main__':
    app.run(debug=True)



