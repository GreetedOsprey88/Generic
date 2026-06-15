import json

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


def build_team_lookup():

    global team_lookup

    team_lookup = {}

    teams = load_teams()

    for team in teams:

        team_lookup[team["team"]] = {
            "rank": int(team["fifa_rank"]),
            "confederation": team["confederation"]
        }


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

    return result, confidence


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/match_predictor", methods=["GET", "POST"])
def match_predictor():

    teams = sorted(team_lookup.keys())

    prediction = None
    confidence = None

    if request.method == "POST":

        selected_team1 = request.form.get("team1")
        selected_team2 = request.form.get("team2")

        if selected_team1 and selected_team2:
            prediction, confidence = predict_match(
                selected_team1,
                selected_team2
            )

    return render_template(
        "match_predictor.html",
        teams=teams,
        prediction=prediction,
        confidence=confidence
    )


@app.route(
    "/wc_predictor",
    methods=["GET", "POST"]
)
def wc_predictor():

    return "World Cup Predictor Coming Soon"


train_match_model()

if __name__ == "__main__":
    app.run(debug=True)