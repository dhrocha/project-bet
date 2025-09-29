import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("./data/league_1.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df = df.sort_values('Date')


def rolling_stat(team_name, current_date, stat, n=10):
    last_games = df[
        ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) &
        (df['Date'] < current_date)
        ].sort_values('Date').tail(n)
    values = []
    for _, row in last_games.iterrows():
        if row['HomeTeam'] == team_name: #Full Time Home Goals
            values.append(row[stat])
        elif row['AwayTeam'] == team_name: #Full Time Away Goals
            values.append(row[stat])

        return np.mean(values) if len(values) > 0 else 0


stats = [
        'FTHG', #FT Home Goals
        'FTAG', #FT Away Goals
        'FTR', #FT Result
        'HTHG', #HT Home Goals
        'HTAG', #HT Away Goals
        'HTR', #HT Result
        'HS', #Home shots
        'AS', #Away Shots
        'HST', #Home SOT
        'AST', #Away SOT
        'HF', #Home Fouls
        'AF', #Away Fouls
        'HC', #Home Corners
        'AC', #Away Corners
        'HY', #Home Yellows
        'AY', #Away Yellows
        'HR', #Home Reds
        'AR' #Away Reds
    ]

home_stats = [
        'FTHG',  # FT Home Goals
        'HTHG',  # HT Home Goals
        'HS',  # Home shots
        'HST',  # Home SOT
        'HF',  # Home Fouls
        'HC',  # Home Corners
        'HY',  # Home Yellows
        'HR',  # Home Reds
    ]

away_stats = [
        'FTAG',  # FT Away Goals
        'HTAG',  # HT Away Goals
        'AS',  # Away shots
        'AST',  # Away SOT
        'AF',  # Away Fouls
        'AC',  # Away Corners
        'AY',  # Away Yellows
        'AR',  # Away Reds
    ]

def add_avg(df, n=10):
    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]

        for stat in home_stats:
                df.loc[idx, f"{stat}_avg"] = rolling_stat(home, date, stat, n)

        for stat in away_stats:
                df.loc[idx, f"{stat}_avg"] = rolling_stat(away, date, stat, n)

    return df

df = add_avg(df)


def preparar_jogo(home, away, date, n=5):
    date = pd.to_datetime(date)

    data = {}
    for stat in home_stats:
        data[f"{stat}_avg"] = rolling_stat(home, date, stat, n)

    for stat in away_stats:
        data[f"{stat}_avg"] = rolling_stat(away, date, stat, n)

    return pd.DataFrame([data])

# ambos marcam
df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)

# ambos marcam ht
df["BTTSHT"] = ((df["HTHG"] > 0) & (df["HTAG"] > 0)).astype(int)

# over 2.5 gols
df['Over25'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)

df['Over15'] = ((df['FTHG'] + df['FTAG']) > 1.5).astype(int)

# over 5 escanteios
df['Over5_Corners'] = ((df['HC'] + df['AC']) > 5).astype(int)

# over 3 cartões amarelos
df['Over3_Yellows'] = ((df['HY'] + df['AY']) > 3).astype(int)

#ambos recebem cartao
df['BTTRC'] = (((df['HY'] + df['HR']) > 0) & ((df['AY'] + df['AR']) > 0)).astype(int)

def train_model(target):
    features = [col for col in df.columns if '_avg' in col]  # todas as médias
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{target} -> Acurácia: {acc:.2f}")

    return model

model_btts = train_model('BTTS')
model_bttsht = train_model('BTTSHT')
model_over25 = train_model('Over25')
model_over15 = train_model('Over15')
model_corners = train_model('Over5_Corners')
model_yellows = train_model('Over3_Yellows')
model_bttrc = train_model('BTTRC')


new_game = preparar_jogo("AFC Wimbledon", "Wycombe", "2025-09-27")

pred_btts = model_btts.predict_proba(new_game)
pred_bttsht = model_btts.predict_proba(new_game)
pred_over_25 = model_over25.predict_proba(new_game)
pred_over_15 = model_over15.predict_proba(new_game)
pred_over_5_corners = model_corners.predict_proba(new_game)
pred_over_3_cards =  model_yellows.predict_proba(new_game)
pred_both_receive_cards = model_bttrc.predict_proba(new_game)

print("Probabilidades:")
print("BTTS:", pred_btts, " | Odd justa:", 1/pred_btts)
print("BTTSHT:", pred_bttsht, " | Odd justa:", 1/pred_bttsht)
print("Over 2.5 gols:", pred_over_25, " | Odd justa:", 1/pred_over_25)
print("Over 1.5 gols:", pred_over_15, " | Odd justa:", 1/pred_over_15)
print("Over 5 escanteios:", pred_over_5_corners, " | Odd justa:", 1/pred_over_5_corners)
print("Over 3 amarelos:", pred_over_3_cards, " | Odd justa:", 1/pred_over_3_cards)
print("Ambos recebem cartao", pred_both_receive_cards, " | Odd justa:", 1/pred_both_receive_cards)
