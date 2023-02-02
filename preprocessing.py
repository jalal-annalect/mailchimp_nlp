
# coding: utf-8
"""
Modified by Jalal Abboud 

Created on Sun Jul 30 12:32:59 2017

@author: DIP
@Copyright: Dipanjan Sarkar
"""

# # Import necessary dependencies

import re
import nltk
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

stopword_list = nltk.corpus.stopwords.words('english')

ps = PorterStemmer()
wnl = WordNetLemmatizer()

# nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)

# # Remove negations from NLTK stopword list
# stopword_list.remove('no')  
# stopword_list.remove('not')



# # Word Tokenization
def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

# # Cleaning Text - strip HTML
def strip_html_tags(text):
    return re.sub(pattern=r'<.*?>', repl=r" ", string=text)

# # Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# # Expanding Contractions
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
                                   if contraction_mapping.get(match) \
                                    else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# # Removing Special Characters and Optionally Digits
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

# # Porter Ptemmer
def stem_text(text):
    tokens = tokenize_text(text)
    text = ' '.join([ps.stem(token) for token in tokens])
    return text

# # Lemmatizing text 
def lemmatize_text(text):
    # Use Spacy to add POS tags for better lemmatization
        # text = nlp(text)  # augment text with POS tags 
        # text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        # return text
    tokens = tokenize_text(text)
    text = ' '.join([wnl.lemmatize(token) for token in tokens])
    return text

# # Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# remove urls 
def remove_urls(text):
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    return text

# remove newlines
def remove_newline(text):
    text = re.sub('/n|\n',' ', text)
    return text

# # Normalize text corpus - tying it all together
def normalize_corpus(corpus,
        html_stripping=False, contraction_expansion=False,
        accented_char_removal=False, remove_digits=False,
        text_lower_case=False, text_lemmatization=False,
        text_stemming=False, special_char_removal=False,
        stopword_removal=False, url_removal=False):
    
    normalized_corpus = []
    
    for doc in corpus:
            
        try:        
            
            doc = re.sub('/n|\n', ' ', doc)

            if url_removal:
                doc = remove_urls(doc)

            if html_stripping:
                doc = strip_html_tags(doc)

            if accented_char_removal:
                doc = remove_accented_chars(doc)

            if contraction_expansion:
                doc = expand_contractions(doc)

            if text_lower_case:
                doc = doc.lower()

            # # Insert spaces between special characters to isolate them    
            # special_char_pattern = re.compile(r'([{.(-)!}])')
            # doc = special_char_pattern.sub(" \\1 ", doc)

            if text_lemmatization:
                doc = lemmatize_text(doc)

            if text_stemming:
                doc = stem_text(doc)

            if special_char_removal:
                doc = remove_special_characters(doc, remove_digits)  

            if stopword_removal:
                doc = remove_stopwords(doc, is_lower_case=text_lower_case)
    
            # padding punctuation with white spaces   
            doc = re.sub('([.,!?()])', r'\1 ', doc) # EX: Hello,world ----> Hello, world
            doc = re.sub(' ([.,!?()]) ', r'\1 ', doc) # EX: Hello , world ----> Hello, world 
            
            # removing extra spaces
            doc = re.sub('/t|\t|/s|\s',' ', doc)
            doc = re.sub(' +', ' ', doc)
            doc = doc.strip()

            normalized_corpus.append(doc)

        except Exception as e:
            print(e)
            continue
        
    return normalized_corpus
