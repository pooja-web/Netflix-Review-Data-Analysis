import pandas as pd
import re
from deep_translator import GoogleTranslator
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import spacy
import nltk


# Load CSV file
df = pd.read_csv("net.csv", sep=None, engine="python")

#before cleaning
print(df)


# -------------------- Cleaning Steps --------------------

# 1. Remove duplicate rows (if any)
df = df.drop_duplicates()

# 2. Handle missing values
df["Rating"] = df["Rating"].fillna(df["Rating"].mean())   # Fill missing ratings
df["ReviewText"] = df["ReviewText"].fillna("No Review")   # Fill missing text reviews

# 3. Standardize column text
df["UserName"] = df["UserName"].str.strip()
df["MovieTitle"] = df["MovieTitle"].str.strip()

# 4. Fix ratings outside valid range (1–5)
df.loc[df["Rating"] > 5, "Rating"] = 5
df.loc[df["Rating"] < 1, "Rating"] = 1

# 5. Convert Date column to datetime format (month/day/year)
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")

# 6. Remove rows with invalid dates
df = df.dropna(subset=["Date"])

# 7. Clean abnormal reviews (remove purely numeric or too short reviews)
df = df[~df["ReviewText"].str.isnumeric()]       # removes reviews like "3333"
df = df[df["ReviewText"].str.len() > 5]          # removes too short reviews

# 8. Capitalize movie titles properly
df["MovieTitle"] = df["MovieTitle"].str.title()

# 9. Standardize usernames (replace spaces with "_")
df["UserName"] = df["UserName"].str.replace(" ", "_")

# 10. Cleaned version (keep for analysis)
def clean_text(text):
    return re.sub(r"[^\w\s]", "", str(text))  # works for all languages

df["ReviewText_Cleaned"] = df["ReviewText"].apply(clean_text).str.strip()

# --------------------------------------------------------

print("\nAfter Cleaning:")
print(df)
# Save cleaned file back to Excel
df.to_csv("netflix_reviews_cleaned.csv", index=False)

#--------------translate-----------

# 11. Translate reviews into English
def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text  # fallback

df["ReviewText_English"] = df["ReviewText"].apply(translate_to_english)

# --------------------------------------------------------

print("\nAfter Cleaning + Translation:")
print(df[["UserName", "MovieTitle", "Rating", "ReviewText", "ReviewText_English"]])

#save after translate the review
df.to_csv("netflix_reviews_translate.csv", index=False)


#-------------Sentiment Analysis--------------

def get_sentiment(text):
     analysis = TextBlob(text).sentiment.polarity
     if analysis > 0:
         return "Positive"
     elif analysis < 0:
         return "Negative"
     else:
         return "Neutral"
    


df["Sentiment_reviewtext"] = df["ReviewText_English"].apply(get_sentiment)



#print(df["Sentiment"])
print("\nAfter Cleaning + Translation + Sentiment_reviewtext:")
print(df[["UserName", "MovieTitle", "Rating", "ReviewText", 
          "ReviewText_English", "Sentiment_reviewtext"]])

# Save final version with sentiment
df.to_csv("sentiment_analysis.csv", index=False)

#-----------Keyword Extraction (Most frequent words or phrases in reviews)-------

all_words = " ".join(df["ReviewText_English"]).lower().split()
common_words = Counter(all_words).most_common(20)
print(common_words)

#---------------- top 5 movies rating---------------
import matplotlib.pyplot as plt

# Group by movie and calculate average rating
avg_ratings = df.groupby("MovieTitle")["Rating"].mean()

# Sort movies by rating descending and take top 5
top5_movies = avg_ratings.sort_values(ascending=False).head(5)
print("Top 5 Highest Rated Movies:")
print(top5_movies)

# Bar chart for top 5 movies
plt.figure(figsize=(10,6))
top5_movies.plot(kind='bar', color='orange')
plt.title("Top 5 Highest Rated Movies")
plt.ylabel("Average Rating")
plt.xlabel("Movie Title")
plt.ylim(0, 5)  # Ratings range 1-5
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#-----------Rating vs. Sentiment Analysis---------

def check_consistency(row):
    if row["Rating"] >= 4:
        rating_sentiment = "Positive"
    elif row["Rating"] <= 2:
        rating_sentiment = "Negative"
    else:
        return "Neutral"

    if rating_sentiment == row["Sentiment_reviewtext"]:
        return "Consistent"
    else:
        return "Mismatch"



# Apply function and create new column
df["RatingVsSentimentAnalysis"] = df.apply(check_consistency, axis=1)

print(df[["UserName","MovieTitle","Rating","ReviewText_English","Sentiment_reviewtext","RatingVsSentimentAnalysis"]])

#------------NER (Named Entity Recognition)-----------



nlp = spacy.load("en_core_web_sm")

def extract_entities(text, fallback):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["WORK_OF_ART", "ORG", "PERSON"]]
    return ", ".join(entities) if entities else fallback

df["Entities"] = df.apply(lambda row: extract_entities(row["ReviewText_English"], row["MovieTitle"]), axis=1)

print("\nAfter NER:")
print(df[["UserName", "MovieTitle", "Rating", "ReviewText_English", "Entities"]])


# Save to Excel or CSV
df.to_csv("reviews_with_entities_NER.csv", index=False)

# -------------------- Save Final Report --------------------
df.to_csv("netflix_reviews_full_report.csv", index=False)





