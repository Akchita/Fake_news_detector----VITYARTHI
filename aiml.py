import os

if not os.path.exists("news.csv"):
    print("Creating dataset...")

    import pandas as pd

    data = {
        "text": [
            # Politics
            "Government launches new education policy",
            "Prime minister announces new scheme",
            "Election results declared today",
            "Parliament passes new bill",
            "New tax reforms introduced",
            "Aliens landed in Delhi last night",
            "Secret government mind control program exposed",
            "Politician caught using magic to win votes",
            "Government hiding alien technology",
            "Country sold to another nation rumor",

            # Science
            "Scientists discover new planet",
            "New vaccine developed for virus",
            "Space mission reaches Mars successfully",
            "Researchers find cure for rare disease",
            "New technology improves battery life",
            "Drinking water cures all diseases instantly",
            "Humans can become invisible says study",
            "Magic pill increases IQ overnight",
            "Time travel machine invented by scientist",
            "Scientists confirm ghosts exist",

            # Social Media
            "Social media platform updates privacy policy",
            "New smartphone model launched",
            "Tech company releases software update",
            "New AI tool helps students learn faster",
            "Cybersecurity update prevents hacking",
            "WhatsApp message says world will end tomorrow",
            "Viral post claims free money for everyone",
            "Instagram hack gives unlimited followers",
            "Facebook deleting all accounts rumor",
            "Click this link to win lottery instantly",

            # Economy
            "Stock market rises due to growth",
            "Bank introduces digital payment system",
            "Economy shows signs of recovery",
            "New startup raises funding",
            "Gold prices increase globally",
            "ATM will stop working forever rumor",
            "Internet will shut down globally rumor",
            "Currency will be banned tomorrow",
            "Bank collapse rumor spreads",
            "Hidden treasure found in city rumor",

            # Sports
            "New sports league announced",
            "Team wins championship final",
            "Player scores record-breaking goals",
            "Olympics preparation begins",
            "New coach appointed for team",
            "Player uses magic to win match",
            "Match fixed using supernatural powers",
            "Team disqualified unfairly rumor",
            "Ghost seen in stadium during match",
            "Player turns invisible during game",

            # Entertainment
            "New movie releases worldwide",
            "Actor wins award for performance",
            "Film breaks box office records",
            "Director announces new project",
            "Music album tops charts",
            "Actor is secretly an alien rumor",
            "Movie causes people to disappear rumor",
            "Hidden messages control minds in songs",
            "Fake celebrity death news spreads",
            "Actor has superpowers in real life",

            # Weather
            "Weather department predicts rain",
            "Heavy rainfall expected this week",
            "Temperature rises during summer",
            "Storm warning issued by authorities",
            "Climate report published",
            "Weather controlled by secret machines",
            "Sun will not rise tomorrow rumor",
            "Artificial rain used to control people",
            "Moon turning red permanently rumor",
            "Earth stopping rotation tomorrow rumor",

            # Education
            "School introduces new curriculum",
            "University announces new courses",
            "Students perform well in exams",
            "Education board releases results",
            "Scholarship program launched",
            "Exam answers leaked everywhere rumor",
            "Students pass without studying rumor",
            "Books contain mind control signals",
            "Fake degree distribution online rumor",
            "Education system hacked completely rumor",

            # Healthcare
            "Healthcare system improves services",
            "Hospital introduces new equipment",
            "Doctors perform successful surgery",
            "New health awareness campaign launched",
            "Medical research published",
            "Doctors hiding cure for profit rumor",
            "Medicine causes instant death rumor",
            "Hospital replaces patients with robots",
            "Virus spread panic fake news",
            "Vaccines contain tracking chips rumor",

            # Transport
            "Transportation system upgraded",
            "New metro line opens",
            "Electric vehicles gaining popularity",
            "Airline launches new routes",
            "Traffic management improved",
            "Cars stop working tomorrow rumor",
            "Planes disappearing mysteriously rumor",
            "Roads built using secret technology rumor",
            "Vehicles run without fuel rumor",
            "Teleportation replacing transport rumor"
        ],

        "label": [
            # Politics (5 real + 5 fake)
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Science
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Social Media
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Economy
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Sports
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Entertainment
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Weather
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Education
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Healthcare
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake",

            # Transport
            "real","real","real","real","real",
            "fake","fake","fake","fake","fake"
        ]
    }

    df_sample = pd.DataFrame(data)
    df_sample.to_csv("news.csv", index=False)

    print("✅ Dataset created successfully!")

 #importing libraries

import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


#loading dataset

df = pd.read_csv("news.csv")

print("\nDataset Loaded Successfully!\n")
print("Total rows:", len(df))


#text cleaning


def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    
    return " ".join(words)

df['text'] = df['text'].apply(clean_text)


#splitting the data( very important)


X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# TF-IDF (IMPROVED)


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)   
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# BETTER MODEL


model = MultinomialNB()   
model.fit(X_train_vec, y_train)


# ACCURACY


y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\n Improved Model Accuracy:", round(accuracy * 100, 2), "%")


# USER INPUT


while True:
    print("\n===== FAKE NEWS DETECTOR =====")
    
    user_input = input("Enter news (or 'exit'): ")
    
    if user_input.lower() == "exit":
        break

    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    
    pred = model.predict(vec)[0]

    if pred == "fake":
        print("❌ FAKE NEWS")
    else:
        print("✅ REAL NEWS")
