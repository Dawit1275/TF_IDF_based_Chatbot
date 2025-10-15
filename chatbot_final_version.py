import requests
from goose3 import Goose
import urllib3
import numpy as np
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer #used to count how many times text appears in a sentence and put it in to matrcies of numbers
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
url = "https://en.wikipedia.org/wiki/Natural_language_processing"
response = requests.get(url,verify=False)
html = response.text
g = Goose()

article_before_preprocessed = g.extract(raw_html=html)
nlp = spacy.load('en_core_web_sm')
original_sentence = [sentence for sentence in nltk.sent_tokenize(article_before_preprocessed.cleaned_text)]

#text preprocessing
def preprocessing(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace('[',' ') #removing a specfic character
    token = [token.text for token in nlp(sentence) if not (token.is_stop or len(token)==1 or token.is_space or token.like_num or token.is_punct)]
    final_token= ' '.join(word for word in token)
    return final_token

def answer(user_text,threshold =0.4):
    cleaned_sentence = []
    for sentence in original_sentence:
        cleaned_sentence.append(preprocessing(sentence)) #appending preprocessed sentences to the cleaned sentences list

    chatbot_answer = ''
    user_text = preprocessing(user_text)
    cleaned_sentence.append(user_text) #appending user input to the cleaned sentences
    TF_IDF= TfidfVectorizer()
    sentence = TF_IDF.fit_transform(cleaned_sentence)
    sentence_similarity = cosine_similarity(sentence[-1].reshape(1, -1),sentence)  # Reshape to 2D array for cosine similarity
    sentence_index = sentence_similarity.argsort()[0][-2]  # Get the index of the most similar sentence (excluding the last one)
    if sentence_similarity[0][sentence_index] > threshold:
        chatbot_answer += original_sentence[sentence_index]
    else:
        chatbot_answer += "I am not sure about that. Can you please rephrase your question?" 
    return chatbot_answer

# #interactive chatbot
import random
welcome_words_input =  ("hi", "hello", "hey", "greetings", "salutations")
welcome_words_output = ("hi there!", "hello!", "hey!", "greetings!", "how are you doing?")

def welcome_message(text):
    found = False  # Initialize found
    for word in text.split():
        if word.lower() in welcome_words_input:
            found = True
    if found:
        return random.choice(welcome_words_output)        

print("Welcome to the chatbot! Type 'exit', 'quit', or 'stop' to end the conversation.")
while True:
    user_text = input("Dear User,How can I help you?:")
    if user_text not in ["exit", "quit", "stop"]:
        if welcome_message(user_text)!= None:
            print("chatbot: " + welcome_message(user_text))
        else:
            print("chatbot: ")
            print(answer(user_text))
    else:
        print("Goodbye! Have a great day!")
        break





