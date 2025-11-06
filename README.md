# Netflix Review Data Analysis and NLP Project
Data Cleaning, Translation, Sentiment Analysis, and NLP insights on Netflix user reviews using Python.

# Project Overview
This project performs data cleaning, translation, sentiment analysis, and NLP-based insights on Netflix user reviews using Python. The goal is to understand user sentiment, highlight top-rated shows, and extract meaningful entities (like movie names, people, or organizations) from text reviews. The entire workflow transforms raw text data into actionable insights through a combination of data preprocessing, translation, machine learning, and visualization techniques.

# Steps Performed
1. **Data Cleaning**- Removed duplicates, filled missing values, standardized text, and fixed invalid ratings/dates.
2. **Translation** - Translated reviews into English using Deep Translator.
3. **Sentiment Analysis** - Used TextBlob to classify reviews as Positive, Negative, or Neutral.
4. **Keyword Extraction** - Found most frequent words using Python's Counter.
5. **Visualization** - Displayed Top 5 Highest Rated Movies using Matplotlib.
6. **Named Entity Recognition (NER)** - Extracted entities (movies, people, orgs) using SpaCy.

### 🧠 Libraries Used
- pandas
- numpy
- matplotlib
- textblob
- deep_translator
- spacy
- nltk

# Outputs
- Cleaning Data Output :
  <img width="607" height="245" alt="image" src="https://github.com/user-attachments/assets/f5430dda-cb31-47aa-8621-d2fd50949188" />

- Translation output :
  <img width="624" height="191" alt="image" src="https://github.com/user-attachments/assets/745cb6b6-b6a3-4771-9d3c-b54f1f99eeb7" />
  
- Sentiment Analysis Output:
  <img width="624" height="220" alt="image" src="https://github.com/user-attachments/assets/4687de9d-88f7-405b-907d-4edd2e5f7386" />

- Keyword Extraction Output:
  <img width="624" height="88" alt="image" src="https://github.com/user-attachments/assets/5515ef08-e587-467d-83af-3e7fb6972af1" />

- Top 5 Highest Rating Output:
  <img width="275" height="128" alt="image" src="https://github.com/user-attachments/assets/d994bfdd-5481-4a54-bae2-a642f1734331" />
  <img width="498" height="208" alt="image" src="https://github.com/user-attachments/assets/37305933-57c5-4ea2-9855-1d2acd38e74a" />

- Rating vs Sentiment Consistency Check output :
  <img width="624" height="182" alt="image" src="https://github.com/user-attachments/assets/b5dc5ea1-4a5d-4020-a5ce-28038fe7318c" />

-  Named Entity Recognition (NER) Output:
  <img width="624" height="201" alt="image" src="https://github.com/user-attachments/assets/22dbbc3c-8c55-4e8e-9774-acabfc678305" />

# Files Included
| File | Description |
| `net.csv` | Raw dataset |
| `netflix_reviews_cleaned.csv` | Cleaned dataset |
| `netflix_reviews_translate.csv` | Translated dataset |
| `sentiment_analysis.csv` | Final dataset with sentiment |
| `netflix_analysis.py` | Python script for analysis |
| `Netflixreviewreport.docx` | Project report documentation |


# Conclusion
This project demonstrates an end-to-end data analysis pipeline integrating:
Data Cleaning
Language Translation
Sentiment Analysis
Keyword Extraction
NER
Visualization
It provides valuable insights into audience perception of Netflix content and can be scaled to handle larger datasets for automated brand or product review analysis.

 # Author: P Pooja 
 # Email : pooja97414@gmail.com
 # linkedin : 

