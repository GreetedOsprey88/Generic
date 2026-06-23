import json
import pickle
import os

from flask import Flask
from flask import render_template
from flask import request

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

match_model = HistGradientBoostingClassifier(
    max_iter=250,
    learning_rate=0.05,
    random_state=42
)

team_lookup = {}
model_accuracy = 0

def clean_obj(obj):
    if isinstance(obj, dict):
        return {k: clean_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_obj(i) for i in obj]
    elif isinstance(obj, str):
        return obj.encode('ascii', 'ignore').decode()
    else:
        return obj

def load_teams():

    with open(
        "Final Project/wc_2026_teams.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_matches():

    with open(
        "Final Project/wc_all_matches.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)

def load_2026_matches():

    with open(
        "Final Project/wc_2026_matches.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_team_lookup():

    global team_lookup

    team_lookup = {}

    teams = load_teams()

    for team in teams:

        team_lookup[team["team"]] = {
            "rank": int(team["fifa_rank"]),
            "confederation": team["confederation"]
        }


MODEL_FILE = "match_model.pkl"

def save_model():
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "model": match_model,
            "team_lookup": team_lookup,
            "accuracy": model_accuracy
        }, f)


def load_model():
    global match_model
    global team_lookup
    global model_accuracy

    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)

    match_model = data["model"]
    team_lookup = data["team_lookup"]
    model_accuracy = data["accuracy"]

def train_match_model():

    global model_accuracy

    build_team_lookup()

    matches = load_matches()

    X = []
    y = []

    for match in matches:

        team1 = match["team1"]
        team2 = match["team2"]

        if team1 not in team_lookup:
            continue

        if team2 not in team_lookup:
            continue

        score1 = int(match["score1"])
        score2 = int(match["score2"])

        rank1 = team_lookup[team1]["rank"]
        rank2 = team_lookup[team2]["rank"]

        same_confederation = int(
            team_lookup[team1]["confederation"]
            ==
            team_lookup[team2]["confederation"]
        )

        X.append([
            rank1,
            rank2,
            rank1 - rank2,
            same_confederation
        ])

        if score1 > score2:
            y.append(0)

        elif score2 > score1:
            y.append(1)

        else:
            y.append(2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    match_model.fit(X_train, y_train)

    predictions = match_model.predict(X_test)

    model_accuracy = round(
        accuracy_score(y_test, predictions) * 100,
        2
    )

    print()
    print("Model trained successfully")
    print("Accuracy:", model_accuracy, "%")
    print()
    
    save_model()


def predict_match(team1, team2):

    rank1 = team_lookup[team1]["rank"]
    rank2 = team_lookup[team2]["rank"]

    same_confederation = int(
        team_lookup[team1]["confederation"]
        ==
        team_lookup[team2]["confederation"]
    )

    features = [[
        rank1,
        rank2,
        rank1 - rank2,
        same_confederation
    ]]

    prediction = match_model.predict(features)[0]

    probabilities = match_model.predict_proba(features)[0]

    confidence = round(
        max(probabilities) * 100,
        2
    )

    if prediction == 0:
        result = f"{team1} Win"

    elif prediction == 1:
        result = f"{team2} Win"

    else:
        result = "Draw"

    return result, confidence, prediction


@app.route("/")
def home():

    return render_template(
        "index.html"
    )

@app.route("/wc_predictor_loading")
def wc_predictor_loading():
    return render_template("wc_loading.html")

@app.route("/match_predictor", methods=["GET", "POST"])
def match_predictor():

    teams = sorted(team_lookup.keys())

    prediction = None
    confidence = None
    error = None
    selected_team1 = None
    selected_team2 = None

    if request.method == "POST":

        selected_team1 = request.form.get("team1")
        selected_team2 = request.form.get("team2")

        # Prevent same team selection
        if selected_team1 == selected_team2:
            error = "Please select two different teams."

        elif selected_team1 and selected_team2:

            prediction, confidence, a = predict_match(
                selected_team1,
                selected_team2
            )

            # uncertainty

            if confidence is not None and 40 <= confidence <= 60:
                prediction = "This match is too close to call (tie/uncertain)."

    return render_template(
        "match_predictor.html",
        teams=teams,
        prediction=prediction,
        confidence=confidence,
        error=error,
        selected_team1=selected_team1,
        selected_team2=selected_team2
    )
@app.route("/process")

def process():
    
    with open("Final Project/wc_2026_matches.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = clean_obj(data)

    with open("Final Project/wc_2026_matches.json", 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    return "jumpScare!"


@app.route("/retrain")
def retrain():

    train_match_model()

    return (
        f"Model retrained successfully. "
        f"Accuracy: {model_accuracy}%"
    )
@app.route(
    "/wc_predictor",
    methods=["GET", "POST"]
)
def wc_predictor():
    # this is for the world cup group stage, I think I am having a bit of problemts but that is easily fixeddd. Check the error message. I don't know really how your ml works... Sry :(
    teams = load_teams()
    group_stage= [{"A" : []}, 
                  {"B" : []}, 
                  {"C" : []}, 
                  {"D" : []}, 
                  {"E" : []}, 
                  {"F" : []}, 
                  {"G" : []}, 
                  {"H" : []},
                  {"I" : []},
                  {"J" : []},
                  {"K" : []},
                  {"L" : []},
                 ]
    knockout_stage = [
        {"Round of 32": []},
        {"Round of 16" : []},
        {"Quarter-final": []},
        {"Semi-final" : []},
        {"Final" : []},
        {"3rd Place Match" : []},
        {"Winner" : ""},
        {"3rd" : ""}
    ]
    #format: {"A": [{"Qatar" : 0}, "Ecuador", "Senegal", "Netherlands"]},  
    for team in teams:
        for group_dict in group_stage:
            group_key = list(group_dict.keys())[0]
            if team["group"] == group_key:
                group_dict[group_key].append({team["team"]: 0})
                break
            
    for match in load_2026_matches():
        if match["stage"] == "Group Stage":    
            

            result, confidence, prediction  = predict_match(
                match["team1"],
                match["team2"]
            )

            
            if match["team2"] == "Canada":
                print(prediction)
            if confidence <= 65:
                #find the teams and both add one to their score. 
                for group_dict in group_stage:
                    group_key = list(group_dict.keys())[0]
                    if match["group"] in group_key:
                        for team_dict in group_dict[group_key]:
                            if match["team1"] in team_dict:
                                team_dict[match["team1"]] += 1
                                print("Team one Have added")
                            if match["team2"] in team_dict:
                                team_dict[match["team2"]] += 1
                                print("team two have been added")
                           
            elif prediction == 0:
                print("Team one Won!")
                for group_dict in group_stage:
                    group_key = list(group_dict.keys())[0]
                    if match["group"] in group_key:
                        for team_dict in group_dict[group_key]:
                            if match["team1"] in team_dict:
                                team_dict[match["team1"]] += 3
                        break
            elif prediction == 1:
                print("Team Two Won!")
                for group_dict in group_stage:
                    group_key = list(group_dict.keys())[0]
                    if match["group"] in group_key:
                        for team_dict in group_dict[group_key]:
                            if match["team2"] in team_dict:
                                team_dict[match["team2"]] += 3
                        break
        
            for group in group_stage:
                group_key = list(group.keys())[0]
                if match["group"] == group_key:
                    teams_in_group = group[group_key]
                    sorted_teams = sorted(teams_in_group, key=lambda x: list(x.values())[0], reverse=True)
                    group[group_key] = sorted_teams
        

        elif match["stage"] == "Round of 32":
            a = list(match["team1"])
            b = list(match["team2"])
            
            for group in group_stage:
                group_key = list(group.keys())[0]
                if group_key == a[1]:
                    team1 = group[group_key][int(a[0]) - 1]
                if group_key == b[1]:
                    team2 = group[group_key][int(b[0]) - 1]           

            knockout_stage[0]["Round of 32"].append({
                "match_id": match["match_id"],
                "stage" : match["stage"],
                "team1": team1,
                "team2": team2,
                
            })

        elif match["stage"] == "Round of 16":
            a = match["team1"].split("-")[2]
            b = match["team2"].split("-")[2]
            for group in knockout_stage[0]["Round of 32"]:
                
                if group["match_id"].split("-")[1] == a:
                    
                    c = list(group["team1"].keys())[0]
                    d = list(group["team2"].keys())[0]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team1 = c
                    elif prediction == 1:
                        team1 = d

                if group["match_id"].split("-")[1] == b:
                    c = list(group["team1"].keys())[0]
                    d = list(group["team2"].keys())[0]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team2 = c
                    elif prediction == 1:
                        team2 = d

            
            knockout_stage[1]["Round of 16"].append({
                "match_id": match["match_id"], # penis <====8 
                "stage" : match["stage"],
                "team1": team1,
                "team2": team2,
                
            })

        elif match["stage"] == "Quarter-final":
            a = match["team1"].split("-")[2]
            b = match["team2"].split("-")[2]
            for group in knockout_stage[1]["Round of 16"]:
                
                if group["match_id"].split("-")[1] == a:
                    print(group["team1"])
                    c =group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team1 = c
                    elif prediction == 1:
                        team1 = d

                if group["match_id"].split("-")[1] == b:
                    c = group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team2 = c
                    elif prediction == 1:
                        team2 = d

            print(team1, team2)
            knockout_stage[2]["Quarter-final"].append({
                "match_id": match["match_id"], # penis <====8 
                "stage" : match["stage"],
                "team1": team1,
                "team2": team2,
                
            })

        elif match["stage"] == "Semi-final":
            a = match["team1"].split("-")[2]
            b = match["team2"].split("-")[2]
            for group in knockout_stage[2]["Quarter-final"]:
                
                if group["match_id"].split("-")[1] == a:
                    print(group["team1"])
                    c =group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team1 = c
                    elif prediction == 1:
                        team1 = d

                if group["match_id"].split("-")[1] == b:
                    c = group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team2 = c
                    elif prediction == 1:
                        team2 = d

            print(team1, team2)
            knockout_stage[3]["Semi-final"].append({
                "match_id": match["match_id"], # penis <====8 
                "stage" : match["stage"],
                "team1": team1,
                "team2": team2,
                
            })
        elif match["stage"] == "Final":
            a = match["team1"].split("-")[2]
            b = match["team2"].split("-")[2]
            for group in knockout_stage[3]["Semi-final"]:
                
                if group["match_id"].split("-")[1] == a:
                    print(group["team1"])
                    c =group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team1 = c
                    elif prediction == 1:
                        team1 = d

                if group["match_id"].split("-")[1] == b:
                    c = group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team2 = c
                    elif prediction == 1:
                        team2 = d

            
            knockout_stage[4]["Final"].append({
                "match_id": match["match_id"], # penis <====8 
                "stage" : match["stage"],
                "team1": team1,
                "team2": team2,
                
            })
        elif match["stage"] == "3rd Place Match":
            a = match["team1"].split("-")[2]
            b = match["team2"].split("-")[2]
            for group in knockout_stage[3]["Semi-final"]:
                
                if group["match_id"].split("-")[1] == a:
                    print(group["team1"])
                    c =group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team1 = d
                    elif prediction == 1:
                        team1 = c

                if group["match_id"].split("-")[1] == b:
                    c = group["team1"]
                    d = group["team2"]
                    result, confidence, prediction  = predict_match(
                        c,
                        d
                    )
                    if prediction == 0:
                        team2 = d
                    elif prediction == 1:
                        team2 = c

            
            knockout_stage[5]["3rd Place Match"].append({
                "match_id": match["match_id"], # penis <====8 
                "stage" : match["stage"],
                "team1": team1,
                "team2": team2,
                
            })
   
    result, confidence, prediction  = predict_match(
        knockout_stage[4]["Final"][0]["team1"],
        knockout_stage[4]["Final"][0]["team2"]
    )
    if prediction == 0:
        knockout_stage[6]["Winner"] = team1
    elif prediction == 1:
        knockout_stage[6]["Winner"] = team2

    result, confidence, prediction  = predict_match(
        knockout_stage[5]["3rd Place Match"][0]["team1"],
        knockout_stage[5]["3rd Place Match"][0]["team2"]
    )
    if prediction == 0:
        knockout_stage[7]["3rd"] = knockout_stage[5]["3rd Place Match"][0]["team1"]
    elif prediction == 1:
        knockout_stage[7]["3rd"] = knockout_stage[5]["3rd Place Match"][0]["team2"]    
    
    return render_template(
    "wc_predictor.html",
    knockout_stage=knockout_stage
)
# in match pred - if conf is +- 10% from 50, match is tied/uncertain
# make sure you can't enter 2 of the same teams



if os.path.exists(MODEL_FILE):
    load_model()
    print(f"Loaded saved model ({model_accuracy}% accuracy)")
else:
    train_match_model()

if __name__ == "__main__":
    app.run(debug=True)