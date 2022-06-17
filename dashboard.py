# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 01:41:46 2022

@author: 60112
"""

import pandas as pd
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import pickle as pkl
from scipy import sparse

# Data Visualization

import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS

# Text Processing
import re
import itertools
import string
import collections
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Machine Learning packages
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.cluster as cluster
from sklearn.manifold import TSNE
import joblib

# Model training and evaluation
from sklearn.model_selection import train_test_split

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

#Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import classification_report

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

#extract lyrics
import lyricsgenius

from PIL import Image
image = Image.open('mbti.jpg')


lemmatiser = WordNetLemmatizer()

# Remove the stop words for speed 
useless_words = stopwords.words("english")

######################


#eda
p2_survey = pd.read_csv('mbtidata.csv')
#p2 = pd.read_csv('../python/p2_lyric1000.csv')
#remove unwanted words in order to get valid model accuracy estimation for unseen data. 
remove_words = '|'.join(['Chorus', 'Lyrics', 'Intro', 'Verse','Outro','Post-Chorus:','Pre-Chorus', 'Embed','Bridge'])
p2_survey["lyrics"] = p2_survey["lyrics"].str.replace(remove_words, '')
#p2_survey.head()
#p2_survey.lyrics[0]

#p2_data = pd.read_csv('../python/p2_NEW.csv')
data = p2_survey[['MBTI','lyrics']]

# Cleaning of data in the lyric
def pre_process_text(data, remove_stop_words=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
      
    for row in data.iterrows():

        #Remove and clean comments
        lyrics = row[1].lyrics

        #Remove Non-words - keep only words
        temp = re.sub("[^a-zA-Z]", " ", lyrics)

        # Remove spaces > 1
        temp = re.sub(' +', ' ', temp).lower()

        #Remove multiple letter repeating words
        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        #Remove stop words
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])


      # transform mbti to binary vector
        #type_labelized = translate_personality(row[1].MBTI) #or use lab_encoder.transform([row[1].type])[0]
        #list_personality.append(type_labelized)
        # the cleaned data temp is passed here
        list_posts.append(temp)

  # returns the result
    list_posts = np.array(list_posts)
    #list_personality = np.array(list_personality)
    return list_posts

list_posts  = pre_process_text(data, remove_stop_words=True)



# Vectorizing the database posts to a matrix of token counts for the model
cntizer = CountVectorizer(analyzer="word", 
                             max_features=770,  
                             max_df=0.7,
                             min_df=0.1) 
# the feature should be made of word n-gram 
# Learn the vocabulary dictionary and return term-document matrix
#print("Using CountVectorizer :")
X_cnt = cntizer.fit_transform(list_posts)

#The enumerate object yields pairs containing a count and a value (useful for obtaining an indexed list)
feature_names = list(enumerate(cntizer.get_feature_names()))
#print("10 feature names can be seen below")
#print(feature_names[0:10])

# For the Standardization or Feature Scaling Stage :-
# Transform the count matrix to a normalized tf or tf-idf representation
tfizer = TfidfTransformer()

# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
#print("\nUsing Tf-idf :")

#print("Now the dataset size is as below")
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
#print(X_tfidf.shape)
#print(X_cnt.shape)
###################


def song_extract(songs):
    
   # songs = recur()
# Log into Genius API with your Authorization Code
    client_access_token='yIyA-7gLpCLUtkU7Udq05X452sNQTNddQdcsRaPeVkz2M_xRuYXwW0pjC7sYu3Nq'
    LyricsGenius = lyricsgenius.Genius(client_access_token)
    
    # The package got some timeout issue so these two lines are needed. If you don't then there will be error when you scrape
    # Source: https://github.com/johnwmillr/LyricsGenius/issues/121
    LyricsGenius.timeout = 15  #timeout
    LyricsGenius.sleep = 5
    
    # Create an array to store each song's lyric
    lyrics_input = []
    lyrics_arr = []
    
    
    for i in songs:
        # get title
        #song_title = ['sza - love galore', 'sza - good days']
        
        # search for song in genius.com
        searched_song = LyricsGenius.search_song(i)
        
        # if we can't find a song's lyrics then skip and append empty string
        if searched_song is None:
            lyrics_input.append("")
            continue
            
        # get the lyric
        lyric = searched_song.lyrics
        
        # replace the lyrics newline with ". "
        lyric = lyric.replace("\n", ". ")
        
        # append the processed lyric to the array
        lyrics_input.append(lyric)
    return lyrics_input

#lyrics_input = song_extract()



def process_input(lyrics_input):
    #process the input
    #lyrics_input = song_extract()
    
    li = ' '.join([str(x) for x in lyrics_input])
    md = pd.DataFrame(data={'MBTI': [''], 'lyrics': [li]})
    
    remove_words = '|'.join(['Chorus', 'Lyrics', 'Intro', 'Verse','Outro','Post-Chorus:','Pre-Chorus', 'Embed','Bridge'])
    md["lyrics"] = md["lyrics"].str.replace(remove_words, '')
    
    li  = pre_process_text(md, remove_stop_words=True)
    my_X_cnt = cntizer.transform(li)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()
    return my_X_tfidf




#To show result output for personality prediction
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
def translate_back(personality):
    # transform binary vector to mbti personality 
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s



#testing with trained model
category = ['0','1','2','3']

def predict(data):

    Dict = []
    for cat in category:
        model = joblib.load(open("XGboost."+ cat +"pkl",'rb'))
        result = model.predict(data)
        Dict.append(result[0])
    return Dict
    #print(f'cat:{cat} result: {result}')
    
#Dict
import streamlit as st


#persoality details
enfj = "Warm, empathetic, responsive, and responsible. Highly attuned to the emotions, needs, and motivations of others. Find potential in everyone, want to help others fulfill their potential. May act as catalysts for individual and group growth. Loyal, responsive to praise and criticism. Sociable, facilitate others in a group, and provide inspiring leadership."
infj = "Seek meaning and connection in ideas, relationships, and material possessions. Want to understand what motivates people and are insightful about others. Conscientious and committed to their firm values. Develop a clear vision about how best to serve the common good. Organized and decisive in implementing their vision."
enfp = "Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information very quickly, and confidently proceed based on the patterns they see. Want a lot of affirmation from others, and readily give appreciation and support. Spontaneous and flexible, often rely on their ability to improvise and their verbal fluency."
entj = "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible. Take care of routine details. Have a clear set of logical standards, systematically follow them and want others to also. Forceful in implementing their plans."
entp = "Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Adept at generating conceptual possibilities and then analyzing them strategically. Good at reading other people. Bored by routine, will seldom do the same thing the same way, apt to turn to one new interest after another."
esfj = "Warmhearted, conscientious, and cooperative. Want harmony in their environment, work with determination to establish it. Like to work with others to complete tasks accurately and on time. Loyal, follow through even in small matters. Notice what others need in their day-by-day lives and try to provide it. Want to be appreciated for who they are and for what they contribute."
esfp = "Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Bring common sense and a realistic approach to their work, and make work fun. Flexible and spontaneous, adapt readily to new people and environments. Learn best by trying a new skill with other people."
estj = "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible. Take care of routine details. Have a clear set of logical standards, systematically follow them and want others to also. Forceful in implementing their plans."
estp = "Flexible and tolerant, they take a pragmatic approach focused on immediate results. Theories and conceptual explanations bore them - they want to act energetically to solve the problem. Focus on the here-and-now, spontaneous, enjoy each moment that they can be active with others. Enjoy material comforts and style. Learn best through doing."
infp = "Idealistic, loyal to their values and to people who are important to them. Want an external life that is congruent with their values. Curious, quick to see possibilities, can be catalysts for implementing ideas. Seek to understand people and to help them fulfill their potential. Adaptable, flexible, and accepting unless a value is threatened."
intj = "Have original minds and great drive for implementing their ideas and achieving their goals. Quickly see patterns in external events and develop long-range explanatory perspectives. When committed, organize a job and carry it through. Skeptical and independent, have high standards of competence and performance - for themselves and others."
intp = "Seek to develop logical explanations for everything that interests them. Theoretical and abstract, interested more in ideas than in social interaction. Quiet, contained, flexible, and adaptable. Have unusual ability to focus in depth to solve problems in their area of interest. Skeptical, sometimes critical, always analytical."
isfj = "Quiet, friendly, responsible, and conscientious. Committed and steady in meeting their obligations. Thorough, painstaking, and accurate. Loyal, considerate, notice and remember specifics about people who are important to them, concerned with how others feel. Strive to create an orderly and harmonious environment at work and at home."
isfp = "Quiet, friendly, sensitive, and kind. Enjoy the present moment, what's going on around them. Like to have their own space and to work within their own time frame. Loyal and committed to their values and to people who are important to them. Dislike disagreements and conflicts, do not force their opinions or values on others."
istj = "Quiet, serious, earn success by thoroughness and dependability. Practical, matter-of-fact, realistic, and responsible. Decide logically what should be done and work toward it steadily, regardless of distractions. Take pleasure in making everything orderly and organized - their work, their home, their life. Value traditions and loyalty."
istp = "Tolerant and flexible, quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily get through large amounts of data to isolate the core of practical problems. Interested in cause and effect, organize facts using logical principles, value efficiency."



 
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: grey;marginTop: -85px'>Welcome to MBTI Predictor</h1>", unsafe_allow_html=True)

cap = "DENTIFICATION OF MYERS-BRIGGS TYPES PERSONALITY (MBTI) BASED ON ONE’S FAVORITE SONGS LYRICS USING TEXT ANALYTICS"

st.sidebar.markdown(f'<p style="color:BLACK;font-weight: bold;font-size:18px;border-radius:2%;">{cap}</p>', unsafe_allow_html=True)
str1="Student name : Izzura Malindo"
str2="Student id : S2024380"

st.sidebar.markdown(f'<p style="color:BLACK;font-weight: bold;font-size:14px;border-radius:2%;">{str1}</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:BLACK;font-weight: bold;font-size:14px;border-radius:2%;">{str2}</p>', unsafe_allow_html=True)
st.sidebar.image(image)

col = st.columns(2)


    

text = st.text_input("Fill in your name",key="name")
    

#text2 =  st.number_input('How many song you wish to inter',min_value=1,max_value=5)
song = [st.text_input("Enter Your song details ..(Follow the format: Singer name - song name ")]
button = st.button("Submit")

if button:
        
        lyrics =  song_extract(song)
        processed_data= process_input(lyrics)
        res = predict(processed_data)
       # st.write(res)
        #st.write(lyrics)
        te= translate_back(res)
        #st.write(te) 
        st.markdown(f'<p style="color:green;font-size:24px;border-radius:2%;">Hi {text}, your MBTI type is {te}</p>', unsafe_allow_html=True)
        
        st.markdown(f'<p style="color:black;font-size:24px;border-radius:2%;">What is your personality type?</p>', unsafe_allow_html=True)
        
        if te == 'ENFJ':
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{enfj}</p>', unsafe_allow_html=True)
        elif te=="INFJ":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{infj}</p>', unsafe_allow_html=True)
        elif te=="ENFP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{enfp}</p>', unsafe_allow_html=True)
        elif te=="ENTJ":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{entj}</p>', unsafe_allow_html=True)
        elif te=="ENTP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{entp}</p>', unsafe_allow_html=True)
        elif te=="ESFJ":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{enfj}</p>', unsafe_allow_html=True)
        elif te=="ESFP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{esfp}</p>', unsafe_allow_html=True)
        elif te=="ESTJ":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{estj}</p>', unsafe_allow_html=True)
        elif te=="ESTP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{estp}</p>', unsafe_allow_html=True)
        elif te=="INFP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{infp}</p>', unsafe_allow_html=True)
        elif te=="INTJ":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{intj}</p>', unsafe_allow_html=True)
        elif te=="INTP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{intp}</p>', unsafe_allow_html=True)
        elif te=="ISFJ":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{isfj}</p>', unsafe_allow_html=True)
        elif te=="ISFP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{isfp}</p>', unsafe_allow_html=True)
        elif te=="ISTJ":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{istj}</p>', unsafe_allow_html=True)
        elif te=="ISTP":
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">{istp}</p>', unsafe_allow_html=True)
        else :
            st.markdown(f'<p style="background-color:#EEB691;color:black;font-size:15px;border-radius:2%;">Nothing found</p>', unsafe_allow_html=True)
        
            
            
      
        
                    
            
    
    
    













