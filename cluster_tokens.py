# coding: utf-8
"""
Created on Wednesday Feb 1 12:32:59 2023

@author: Jalal Abboud
"""

import preprocessing as pp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import spacy
from itertools import chain
import argparse
import warnings

# ** command line arguments

# accept command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-text", "--text", help = "Path to file containing text. Each text should be on a new line.", required=True, type=str)
parser.add_argument("-tokenize", "--tokenize", help = "Bool. whether to tokenize or not.", type=str, required=True)
parser.add_argument("-output", "--output", help = "Directory to output clustered tokens data. Should be a csv or excel file", type=str)
parser.add_argument("-clusters", "--clusters", help = "Number of clusters", type=int, default=5)
parser.add_argument("-language_model", "--language_model", help = "Language model to use. Pass tfidf for term frequency vectorization.", type=str, default="en_core_web_lg")
args = parser.parse_args()

# parsing CML arguments
model = args.language_model
num_clusters = args.clusters
text_path = args.text
output_filename = args.output
tokenize = args.tokenize

# bool parsing from cmd didnt work
if tokenize=="True":
    tokenize=True
else:
    tokenize=False

if model=="tfidf" and tokenize:
    warnings.warn("Tokenization with TF-IDF features will not give any meaningful results.")
else:
    pass

# reading tokens
text = open(text_path, "r").readlines()
print("Cleaning text...")
clean_text = pp.normalize_corpus(text, contraction_expansion=True, accented_char_removal=True, 
                                 remove_digits=True, text_lower_case=True, special_char_removal=True, 
                                 stopword_removal=True, html_stripping=True)
# taking unique clean text
clean_text = list(set(clean_text))

# tokenize if passed
if tokenize:
    print("Tokenizing text...")
    clean_text = [pp.tokenize_text(text) for text in clean_text]
    clean_text = list(set(list(chain(*clean_text))))
else:
    pass

print("Extracting features...")
if model!="tfidf":
    # loading language model
    nlp = spacy.load(model)
    features = [nlp(text).vector for text in clean_text]
else:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_text)
    features = X.toarray()

print("Clustering...")
# clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

print("Saving output...")
# creating dataframe
df = pd.DataFrame(data={"labels":kmeans.labels_, "text":clean_text})
df["Value"] = 1

# pivoting
df = df.pivot(index='text', columns='labels', values='Value').fillna(0)

# saving 
if output_filename.endswith(".csv"):
    df.to_csv(output_filename)

elif output_filename.endswith(".xlsx"):
    df.to_excel(output_filename)

else: 
    raise Exception("Currently not supporting the passed extenstion.")

print("Done!")