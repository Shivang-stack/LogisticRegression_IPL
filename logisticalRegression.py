import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')


matches_filtered = matches[(matches['winner'] == matches['team1']) | (matches['winner'] == matches['team2'])].copy()
matches_filtered['outcome'] = (matches_filtered['winner'] == matches_filtered['team1']).astype(int)

features = matches_filtered[['team1', 'team2']]
target = matches_filtered['outcome']

features_encoded = pd.get_dummies(features, columns=['team1', 'team2'])


X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Model Test Accuracy: {:.2f}%".format(accuracy * 100))

def predict_match(team1, team2, team1_players, team2_players):
    input_df = pd.DataFrame(data={col: [0] for col in features_encoded.columns})
    
    col_team1 = f"team1_{team1}"
    if col_team1 in input_df.columns:
        input_df[col_team1] = 1
    col_team2 = f"team2_{team2}"
    if col_team2 in input_df.columns:
        input_df[col_team2] = 1
    
    prob = model.predict_proba(input_df)[0]
    prediction = {
        team1: round(prob[1] * 100, 2),
        team2: round((1 - prob[1]) * 100, 2)
    }
    
   
    players = team1_players + team2_players
    batter_stats = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
    match_batters = batter_stats[batter_stats['batter'].isin(players)]
    top_batters = match_batters.sort_values(by='batsman_runs', ascending=False).head(5)['batter'].tolist()

   
    valid_wickets = deliveries[deliveries['dismissal_kind'].notnull() & (deliveries['dismissal_kind'] != 'run out')]
    bowler_stats = valid_wickets.groupby('bowler')['dismissal_kind'].count().reset_index()
    bowler_stats.columns = ['bowler', 'wickets']

    match_bowlers = bowler_stats[bowler_stats['bowler'].isin(players)]
    top_bowlers = match_bowlers.sort_values(by='wickets', ascending=False).head(5)['bowler'].tolist()

    return prediction, top_batters, top_bowlers


team1 = 'Kolkata Knight Riders'
team2 = 'Rajasthan Royals'
team1_players =[
    "Quinton de Kock (WK)",
    "Venkatesh Iyer",
    "Ajinkya Rahane (C)",
    "Rinku Singh",
    "Moeen Ali",
    "Andre Russell",
    "Ramandeep Singh",
    "Spencer Johnson",
    "Vaibhav Arora",
    "Harshit Rana",
    "Varun Chakaravarthy"
]
team2_players = [
    "Yashasvi Jaiswal",
    "Sanju Samson",
    "Nitish Rana",
    "Riyan Parag (C)",
    "Dhruv Jurel (WK)",
    "Shimron Hetmyer",
    "Wanindu Hasaranga",
    "Jofra Archer",
    "Maheesh Theekshana",
    "Tushar Deshpande",
    "Sandeep Sharma"
]
prediction, top_batters, top_bowlers = predict_match(team1, team2, team1_players, team2_players)

print("\nMatch Outcome Prediction (Winning Probabilities):")
for team, prob in prediction.items():
    print(f"{team}: {prob}%")

print("\nTop 5 Batters Based on Historical Runs:")
for idx, player in enumerate(top_batters, start=1):
    print(f"{idx}. {player}")

print("\nTop 5 Bowlers Based on Historical Wickets:")
for idx, player in enumerate(top_bowlers, start=1):
    print(f"{idx}. {player}")