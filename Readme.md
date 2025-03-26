# IPL Match Outcome Prediction

## Overview
This project predicts the outcome of an IPL match using a **Logistic Regression model**. Given two teams and their player lists, it:
- Predicts the **winning probability** for each team.
- Identifies the **top 5 batters** based on historical runs.
- Identifies the **top 5 bowlers** based on historical wickets.

## Dataset
Uses two datasets:
1. `matches.csv` – Contains details of past IPL matches, including the winner.
2. `deliveries.csv` – Ball-by-ball data for all IPL matches, used to identify top players.

## Setup
### **1. Install dependencies**
```bash
pip install pandas scikit-learn numpy
```

### **2. Run the script**
```python
python logisticalRegression.py
```

## How It Works
1. **Train the Model**
   - Uses past matches to train a Logistic Regression model.
   - Encodes teams and predicts winning probability based on historical performance.

2. **Predict Match Outcome**
   - User inputs `team1`, `team2`, and their respective `player lists`.
   - The model predicts the probability of each team winning.

3. **Identify Top Players**
   - **Top Batters**: Selected based on total career runs.
   - **Top Bowlers**: Selected based on total wickets taken.

## Example Usage
```python
team1 = 'Kolkata Knight Riders'
team2 = 'Rajasthan Royals'
team1_players = ["Quinton de Kock (WK)",
    "Venkatesh Iyer",
    "Ajinkya Rahane (C)",
    "Rinku Singh",
    "Moeen Ali",
    "Andre Russell",
    "Ramandeep Singh",
    "Spencer Johnson",
    "Vaibhav Arora",
    "Harshit Rana",
    "Varun Chakaravarthy"]
team2_players = [ "Yashasvi Jaiswal",
    "Sanju Samson",
    "Nitish Rana",
    "Riyan Parag (C)",
    "Dhruv Jurel (WK)",
    "Shimron Hetmyer",
    "Wanindu Hasaranga",
    "Jofra Archer",
    "Maheesh Theekshana",
    "Tushar Deshpande",
    "Sandeep Sharma"]

prediction, top_batters, top_bowlers = predict_match(team1, team2, team1_players, team2_players)

print(prediction) 
print(top_batters) 
print(top_bowlers)


Outcome:
Match Outcome Prediction (Winning Probabilities):
Kolkata Knight Riders: 56.9%
Rajasthan Royals: 43.1%

Top 5 Batters Based on Historical Runs:
1. Ramandeep Singh
2. Sandeep Sharma
3. Harshit Rana

Top 5 Bowlers Based on Historical Wickets:
1. Sandeep Sharma
2. Harshit Rana
3. Ramandeep Singh
```

## Next Steps
- Improve model accuracy with more features (e.g., venue, recent form).
- Integrate a web interface to make predictions user-friendly.
