import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# carregar dataset
df = pd.read_csv("./data/football_matches.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")  # ajuste de data
df = df.sort_values('Date')


# função para calcular médias passadas
def rolling_stat(team_name, current_date, stat, n=5):
    jogos_passados = df[
        ((df['Home_Team'] == team_name) | (df['Away_Team'] == team_name)) &
        (df['Date'] < current_date)
    ].sort_values('Date').tail(n)

    valores = []
    for _, row in jogos_passados.iterrows():
        if row['Home_Team'] == team_name and stat.startswith("Home"):
            valores.append(row[stat])
        elif row['Away_Team'] == team_name and stat.startswith("Away"):
            valores.append(row[stat])
        elif row['Home_Team'] == team_name and stat.startswith("Away"):
            valores.append(row[stat.replace("Away","Home")])
        elif row['Away_Team'] == team_name and stat.startswith("Home"):
            valores.append(row[stat.replace("Home","Away")])

    return np.mean(valores) if len(valores) > 0 else 0


def adicionar_medias(df, n=5):
    stats = [
        "Home_Team_Score", "Away_Team_Score",
        "Home_Team_Yellowcard", "Away_Team_Yellowcard",
        "Home_Team_Redcard", "Away_Team_Redcard",
        "Home_Team_Shots", "Away_Team_Shots",
        "Home_Team_ShotsonTarget", "Away_Team_ShotsonTarget",
        "Home_Team_Possession", "Away_Team_Possession",
        "Home_Team_Passes", "Away_Team_Passes",
        "Home_Team_Fouls", "Away_Team_Fouls",
        "Home_Team_Offside", "Away_Team_Offside",
        "Home_Team_Corner", "Away_Team_Corner"
    ]

    for idx, row in df.iterrows():
        home = row["Home_Team"]
        away = row["Away_Team"]
        date = row["Date"]

        for stat in stats:
            if stat.startswith("Home"):
                df.loc[idx, f"{stat}_avg"] = rolling_stat(home, date, stat, n)
            else:
                df.loc[idx, f"{stat}_avg"] = rolling_stat(away, date, stat, n)

    return df


# aplicar no dataset
df = adicionar_medias(df)

def preparar_jogo(home, away, date, n=5):
    date = pd.to_datetime(date)
    stats = [
        "Home_Team_Score", "Away_Team_Score",
        "Home_Team_Yellowcard", "Away_Team_Yellowcard",
        "Home_Team_Redcard", "Away_Team_Redcard",
        "Home_Team_Shots", "Away_Team_Shots",
        "Home_Team_ShotsonTarget", "Away_Team_ShotsonTarget",
        "Home_Team_Possession", "Away_Team_Possession",
        "Home_Team_Passes", "Away_Team_Passes",
        "Home_Team_Fouls", "Away_Team_Fouls",
        "Home_Team_Offside", "Away_Team_Offside",
        "Home_Team_Corner", "Away_Team_Corner"
    ]

    data = {}
    for stat in stats:
        if stat.startswith("Home"):
            data[f"{stat}_avg"] = rolling_stat(home, date, stat, n)
        else:
            data[f"{stat}_avg"] = rolling_stat(away, date, stat, n)

    return pd.DataFrame([data])


# ambos marcam
df["BTTS"] = ((df["Home_Team_Score"] > 0) & (df["Away_Team_Score"] > 0)).astype(int)

# over 2.5 gols
df['Over25'] = ((df['Home_Team_Score'] + df['Away_Team_Score']) > 2.5).astype(int)

df['Over15'] = ((df['Home_Team_Score'] + df['Away_Team_Score']) > 1.5).astype(int)

# over 5 escanteios
df['Over5_Corners'] = ((df['Home_Team_Corner'] + df['Away_Team_Corner']) > 5).astype(int)

# over 3 cartões amarelos
df['Over3_Yellows'] = ((df['Home_Team_Yellowcard'] + df['Away_Team_Yellowcard']) > 3).astype(int)


def treinar_modelo(target):
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


# treinar modelos
model_btts = treinar_modelo('BTTS')
model_over25 = treinar_modelo('Over25')
model_over15 = treinar_modelo('Over15')
model_corners = treinar_modelo('Over5_Corners')
model_yellows = treinar_modelo('Over3_Yellows')

# prever próximo jogo
novo_jogo = preparar_jogo("Girona", "Espanyol", "2025-09-26")

pred_btts = model_btts.predict_proba(novo_jogo)
pred_over_25 = model_over25.predict_proba(novo_jogo)
pred_over_15 = model_over15.predict_proba(novo_jogo)
pred_over_5_corners = model_corners.predict_proba(novo_jogo)
pred_over_3_cards =  model_yellows.predict_proba(novo_jogo)

print("Probabilidades:")
print("BTTS:", pred_btts, " | Odd justa:", 1/pred_btts)
print("Over 2.5 gols:", pred_over_25, " | Odd justa:", 1/pred_over_25)
print("Over 1.5 gols:", pred_over_15, " | Odd justa:", 1/pred_over_15)
print("Over 5 escanteios:", pred_over_5_corners, " | Odd justa:", 1/pred_over_5_corners)
print("Over 3 amarelos:", pred_over_3_cards, " | Odd justa:", 1/pred_over_3_cards)