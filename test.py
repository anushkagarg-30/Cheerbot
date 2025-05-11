import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from string import punctuation
import re
faq =pd.read_csv("chatbot.csv")
faq_quest = faq[['Question_ID', 'Questions']]
faq_answ = faq[['Question_ID', 'Answers']]

contractions_dict = {     
"ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
"could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
"don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
"he'd": "he had", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
"how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I had", "I'd've": "I would have",
"I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it had",
"it'd've": "it would have", "it'll": "it will", "it'll've": "iit will have", "it's": "it is", "let's": "let us",
"ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
"must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
"o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
"sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she had", "she'd've": "she would have", "she'll": "she will",
"she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
"shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that had", "that'd've": "that would have",
"that's": "that is", "there'd": "there had", "there'd've": "there would have", "there's": "there is", "they'd": "they had",
"they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
"they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had", "we'd've": "we would have",
"we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
"what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
"when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
"who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
"why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
"wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
"y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had",
"you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

def expand_contraction(text, contraction_dict):
    contraction_pattern= re.compile('({})'.format('|'.join(contraction_dict.keys())), flags= re.IGNORECASE | re.DOTALL)
    
    def expand_match(contraction):
        match= contraction.group(0)
        first_char= match[0]
        expanded_contraction= contraction_dict.get(match) \
            if contraction_dict.get(match) \
            else contraction_dict.get(match.lower())
        expanded_contraction= expanded_contraction
        return expanded_contraction
        
    expanded_text= contraction_pattern.sub(expand_match, text)
    expanded_text= re.sub("'","", expanded_text)
    return expanded_text

def main_contraction(text):
    text = expand_contraction(text, contractions_dict)
    return text

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stopwords.words('english')

def remove_stopwords(text):
    stop_words= stopwords.words('english')
    
    return ' '.join(c for c in nltk.word_tokenize(text) if c not in stop_words)


from nltk.stem import WordNetLemmatizer

wordnet_lemma = WordNetLemmatizer()

def lemma(text):
    lemmatize_words = [wordnet_lemma.lemmatize(word) for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return ' '.join(lemmatize_words)


def to_lower(text):
    return text.lower()
def remove_number(text):
    output = ''.join(c for c in text if not c.isdigit())
    return output
def remove_punct(text):
    return "".join(c for c in text if c not in punctuation)
def to_strip(text):
    return " ".join([c for c in text.split() if len(c)>2])
def remove_char(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    return text
def remove_duplicate(text):
    text = re.sub("(.)\\1{2,}", "\\1", text)
    return text


faq_quest['prep1']= faq_quest['Questions'].apply(to_lower)
faq_quest['prep2']= faq_quest['prep1'].apply(main_contraction)
faq_quest['prep3']= faq_quest['prep2'].apply(remove_number)
faq_quest['prep4']= faq_quest['prep3'].apply(remove_punct)
faq_quest['prep5']= faq_quest['prep4'].apply(to_strip)
faq_quest['prep6']= faq_quest['prep5'].apply(remove_char)
faq_quest['prep7']= faq_quest['prep6'].apply(remove_duplicate)
faq_quest['prep8']= faq_quest['prep7'].apply(remove_stopwords)
faq_quest['lemma']= faq_quest['prep8'].apply(lemma)
faq_quest.head(10)

faq_answ['prep1']= faq_answ['Answers'].apply(to_lower)
faq_answ['prep2']= faq_answ['prep1'].apply(main_contraction)
faq_answ['prep3']= faq_answ['prep2'].apply(remove_number)
faq_answ['prep4']= faq_answ['prep3'].apply(remove_punct)
faq_answ['prep5']= faq_answ['prep4'].apply(to_strip)
faq_answ['prep6']= faq_answ['prep5'].apply(remove_char)
faq_answ['prep7']= faq_answ['prep6'].apply(remove_duplicate)
faq_answ['prep8']= faq_answ['prep7'].apply(remove_stopwords)
faq_answ['lemma']= faq_answ['prep8'].apply(lemma)
faq_answ.head(10)


def dictionary(check):
    check = check.str.extractall('([a-zA_Z]+)')
    check.columns = ['check']
    b = check.reset_index(drop=True)
    check = b['check'].value_counts()
    
    dictionary = pd.DataFrame({'word': check.index, 'freq': check.values})
    dictionary.index = dictionary['word']
    dictionary.drop('word', axis = 1, inplace=True)
    dictionary.sort_values('freq', inplace= True, ascending= False)
    
    return dictionary

dictionary_clean = dictionary(faq_quest['lemma'])
dictionary_clean[:30].plot(kind = 'barh',figsize = (10,10))


import matplotlib.pyplot as plt

    

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
faq['AnswersEncode'] = label.fit_transform(faq['Answers'])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


text = faq['Questions']
y= faq['AnswersEncode'].values


tfidf = TfidfVectorizer(use_idf=True, analyzer='word', stop_words='english', token_pattern=r'\b[^\d\W]+\b', ngram_range=(1,2))
X_train = tfidf.fit_transform(text)
print(X_train)

lsvc = LinearSVC(random_state = 2021)
lsvc.fit(X_train, y)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

score = lsvc.score(X_train, y)



search_test = [
    "how to help my child who’s stressed about placements?"
]

search_engine = tfidf.transform(search_test)
result = lsvc.predict(search_engine)


for question in result:
    faq_data = faq.loc[faq.isin([question]).any(axis=1)]
    print("Answer: ", faq_data['Answers'].values)
    