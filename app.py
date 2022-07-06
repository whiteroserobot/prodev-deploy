# %%writefile main.py
#%pip install streamlit_authenticator
#%pip install sqlalchemy
from sqlalchemy import true
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import datetime
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from datetime import date

import base64

import pickle
from pathlib import Path
import streamlit_authenticator as stauth

hide_menu = """
<style>
#MainMenu {
    visibility:visible;
}
footer {
    visibility:visible;
}
footer:after{
    content:'Copyright © 2022: Indispensables';
    display:block;
    position:relative;
    color:red;
    padding:5px;
    top:3px;

}
</style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)

# image = Image.open("Data\gettyimages-866631268-2048x2048.jpg")
# new_image = image.resize((1150, 550))

@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img ="""
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: 1390px 750px;           
        }
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
# from datetime import datetime

# Loading dataframe

    

df = pd.read_csv('out.csv', low_memory=False)
df.dropna(subset=['tweet_clean'], inplace=True)
df['time'] = pd.to_datetime(df['time']).dt.normalize()
# df = df.sample(3000)

time = value=datetime.datetime(year=2022, month=6, day=10, hour=16, minute=30)
# time = t.strftime("%Y-%m-%d")
time = time.date()

# st.set_page_config(layout="wide")
set_bg("/home/mirana/Desktop/python/scraping/prodev deploy/aaa.jpg")
# st.title("The indespensables 2022 Election Analysis")

# horizontal menu
# navigation = option_menu(
#     menu_title=None,
#     options=["Home", "Politics Today", "Presidential Election Prediction"],
#     icons=["eye", "pin", "eye"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
#     styles={
#         "container": {"padding": "0!important"},
#         "icon": {"color": "#fff", },
#         "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "#A23838"},

#         },

# )

with st.sidebar:
    navigation = st.sidebar.selectbox("Main Menu", ["Home", "Politics Today", "Presidential Election Prediction"])
        # icons=['house', 'activity', 'eye-fill'], menu_icon="menu-app", default_index=1)


st.title("The Indispensables 2022 Election Analysis")
if navigation == "Home":
    # st.image(new_image)
    st.write("""
    **_At indispensable, we care about Political current affairs in Kenya.
        We Also provide you with updates on twitter popularity of the Presidential Candidates in the upcoming elections._** 
    """)    
    
if navigation == "Politics Today":

    st.write("""
    **_Elections are going to be held  on 9th August 2022. Here, we seek to show you the trending topics, the changing popularities
    of political parties and politicians_**
    """)
    

    navigate2 = option_menu(
    menu_title=None,
    options=["Trending Topics", "Political Parties", "Political Figures"],
    icons=["activity", "flag-fill", "people-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
    "container": {"padding": "0!important"},
    "icon": {"color": "#fff", },
    "nav-link": {"font-size": "12px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "#A23838"},

    },
    )
    if navigate2 == "Trending Topics":
        
        st.subheader("Dates of interest")
        st.sidebar.subheader("Dates of interest Analysis")
        
        st.write("What dates do you want to get Trending topics in this election cycle?")
        start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)

        if start < time:
            st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')
        
        end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)

        if start < end and start >= time:
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))

            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")
            
            #greater than the start date and smaller than the end date
            mask = (df['time'] > start) & (df['time'] <= end)

            df1 = df.loc[mask]

            # if start_date: 
            # Converting text descriptions into vectors using TF-IDF using Bigram
            tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase = False)
            tfidf_matrix = tf.fit_transform(df1['tweet_clean'])
            total_words = tfidf_matrix.sum(axis=0) 
        
            # Finding the word frequency
            freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
            freq =sorted(freq, key = lambda x: x[1], reverse=True)

            # converting into dataframe 
            bigram = pd.DataFrame(freq)
            bigram.rename(columns = {0:'bigram', 1: 'count'}, inplace = True) 

            # st.subheader("Top trends on social media in the run up to the 2022 elections")
            choice = st.sidebar.selectbox(label = "choose", options = ["Top trends on social media in the run up to the 2022 elections", 'Polarity of sentiments of the electorate heading towards the general election'])
            if choice == "Top trends on social media in the run up to the 2022 elections":
                st.subheader("Top trends on social media in the run up to the 2022 elections")
                # Taking first 20 records
                popular_words  = bigram.head(20)
                popular_words['count'] = ((popular_words['count']/ popular_words['count'].sum()) * 100).round(2)
                names = popular_words['bigram'].to_list()
                counts = popular_words['count'].to_list()
                source = pd.DataFrame({
                    'counts': counts,
                    'names': names
                    })
                # source1 = source.sort_values(by='counts', ascending = True)
                bar_chart = alt.Chart(source).mark_bar().encode(
                    y='counts',
                    # x='names',
                    x=alt.X('names', sort=None),
                    )
            
                text = bar_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    ).encode(text='counts')
            
                plt = (bar_chart + text).properties(height=600)


                st.altair_chart(plt, use_container_width=True)

            if choice == 'Polarity of sentiments of the electorate heading towards the general election':
                st.subheader('Polarity of sentiments of the electorate heading towards the general election') 
                # st.sidebar.subheader(' Polarity of sentiments of the electorate heading towards the general election') 

                st.write("""
                **_The line chart represents the changes in polarity of the electorate as we head toward the 2022
                election. Score of 1 represents very positive, -1 represents very negative, and 0 represents 
                neautral sentiments_** 
                """)

                # "Electorate Sentiment Polarity"

                df2 = df.loc[mask]

                polarity = df2[['time', 'Polarity']]
                polarity = polarity.groupby('time',  as_index=False, sort=False).agg({'Polarity': 'mean'})
                date1 = polarity['time'].to_list()
                polaritys = polarity['Polarity'].to_list()

                source = pd.DataFrame({
                    'polaritys': polaritys,
                    'date': date1
                    })

                line_chart = alt.Chart(source).mark_line().encode(
                    y='polaritys',
                    x='date',
                    )
            
                text = line_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    )
            
                plt = (line_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)

        else:
            st.error('Error: End date must fall after start date.') 

    if navigate2 == "Political Parties":

        # st.sidebar("Dates of interest", fixed = True)
        st.sidebar.subheader("Dates of interest Analysis")

        st.write("What dates do you want to get popularity of various political parties?")

        start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)

        if start < time:
            st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')
        
        end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)

        if start < end and start >= time:
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))

            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")
            
            #greater than the start date and smaller than the end date
            mask = (df['time'] > start) & (df['time'] <= end)

            df1 = df.loc[mask]

            # if start_date: 
            # Converting text descriptions into vectors using TF-IDF using Bigram
            tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase = False)
            tfidf_matrix = tf.fit_transform(df1['tweet_clean'])
            total_words = tfidf_matrix.sum(axis=0) 
        
            # Finding the word frequency
            freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
            freq =sorted(freq, key = lambda x: x[1], reverse=True)

            # converting into dataframe 
            bigram = pd.DataFrame(freq)
            bigram.rename(columns = {0:'bigram', 1: 'count'}, inplace = True) 

            # Taking first 20 records
            bigram = bigram.head(50)

            choice2 = st.sidebar.selectbox(label = "choose", options = ["Social media mentions of political alliances in the 2022 general elections", 'Polarity of sentiments of the electorate heading towards the general election' ])
            if choice2 == "Social media mentions of political alliances in the 2022 general elections":
                st.subheader("Social media mentions of political alliances in the 2022 general elections")

                the_list = ['kwanza', 'azimio',  'roots', 'kwisha', 'assimio', 'uda', 'anc', 'odm', 'ford kenya', 'wiper', 'maendeleo', 'dap', 'chama cha kazi', 'hustler nation']

                def party_finder(string):
                    term_return = 'None'
                    for term in the_list:
                        if term in string:
                            term_return = term
            
                    return term_return

                bigram['party'] = bigram['bigram'].apply(party_finder)

                # presidential aspirants
                words_clean =bigram.replace({'party' : {'kwanza': 'Kenya Kwanza', 'roots': 'Roots Party', 'hustler nation': 'Kenya Kwanza'}})
                words_partys = words_clean.drop('bigram', axis = 1)

                words_party = words_partys.groupby('party',  as_index=False, sort=False).agg({'count': 'sum'})
                words_party.drop(words_party.loc[words_party['party']== 'None'].index, inplace=True)
                words_party['count'] = ((words_party['count']/ words_party['count'].sum()) * 100).round(2)


                # Names and counts
                names = words_party['party'].to_list()
                counts = words_party['count'].to_list()

                source = pd.DataFrame({
                    'counts': counts,
                    'names': names
                    })
                bar_chart = alt.Chart(source).mark_bar().encode(
                    y='counts',
                    x='names',
                    )
            
                text = bar_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    ).encode(text='counts')
            
                plt = (bar_chart + text).properties(height=600)


                st.altair_chart(plt, use_container_width=True)

            if choice2 == 'Polarity of sentiments of the electorate heading towards the general election':
                st.subheader('Polarity of sentiments of the electorate heading towards the general election')

                st.write("""
                **_The line chart represents the changes in polarity of the electorate as we head toward the 2022
                election. Score of 1 represents very positive, -1 represents very negative, and 0 represents 
                neautral sentiments_**
                """)
                
                the_list = ['kwanza', 'azimio',  'roots', 'kwisha', 'assimio', 'uda',  'hustler nation', 'odm']

                def party_finder(string):
                    term_return = 'None'
                    for term in the_list:
                        if term in string:
                            term_return = term
            
                    return term_return
                
                
                df2 = df.loc[mask]
                df3 = df2[['tweet_clean', 'time', 'Polarity']]
                df3['party'] = df3['tweet_clean'].apply(party_finder)

                # presidential aspirants
                words_clean =df3.replace({'party' : {'kwanza': 'Kenya Kwanza', 'roots': 'Roots Party', 'kwisha': 'Kenya Kwanza', 'azimio': 'azimio OKA alliance', 'assimio':'azimio OKA alliance', 'uda': 'Kenya Kwanza', 'hustler nation': 'Kenya Kwanza', 'odm': 'azimio OKA alliance' }})
                words_partys = words_clean.drop('tweet_clean', axis = 1)

                words_party = words_partys.groupby(['time','party'], as_index=False, sort=False).agg({'Polarity': 'mean'})
                words_party.drop(words_party.loc[words_party['party']== 'None'].index, inplace=True)
                date1 = words_party['time'].to_list()
                polaritys = words_party['Polarity'].to_list()
                party = words_party['party'].to_list()

                source = pd.DataFrame({
                    'polaritys': polaritys,
                    'date': date1,
                    'symbol': party
                    })

                line_chart = alt.Chart(source).mark_line().encode(
                    y='polaritys',
                    x='date',
                    color='symbol',
                    strokeDash='symbol',
                    )
            
                text = line_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    )
            
                plt = (line_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)
            
        else:
            st.error('Error: End date must fall after start date.')       


    if navigate2 == "Political Figures":

        st.sidebar.subheader("Dates of interest Analysis")

        st.write("What dates do you want to get popularity of various political figures?")
        start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)

        if start < time:
            st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')
        
        end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)

        if start < end and start >= time:
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))

            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")
            
            #greater than the start date and smaller than the end date
            mask = (df['time'] > start) & (df['time'] <= end)

            df1 = df.loc[mask]

            # if start_date: 
            # Converting text descriptions into vectors using TF-IDF using Bigram
            tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase = False)
            tfidf_matrix = tf.fit_transform(df1['tweet_clean'])
            total_words = tfidf_matrix.sum(axis=0) 
        
            # Finding the word frequency
            freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
            freq =sorted(freq, key = lambda x: x[1], reverse=True)

            # converting into dataframe 
            bigram = pd.DataFrame(freq)
            bigram.rename(columns = {0:'bigram', 1: 'count'}, inplace = True) 

            # Taking first 20 records
            bigram = bigram.head(50)

            choice3 = st.sidebar.selectbox(label = "choose", options = ["Mentions of 2022 general elections Presidential aspirants on social media", "Polarity of sentiments towards various presidential aspirants for Kenyas 2022 general election"])
            if choice3 == "Mentions of 2022 general elections Presidential aspirants on social media":
                st.subheader("Mentions of 2022 general elections Presidential aspirants on social media")

                the_list = ['wajackoyah', 'raila',  'ruto', 'deputy', 'baba']

                def fruit_finder(string):
                    term_return = 'None'
                    for term in the_list:
                        if term in string:
                            term_return = term
            
                    return term_return

                bigram['term'] = bigram['bigram'].apply(fruit_finder)

                words_clean =bigram.replace({'term' : { 'deputy' : 'Dr. Ruto', 'baba' : 'Mr. Odinga', 'ruto' : 'Dr. Ruto', 'raila': 'Mr. Odinga',  'wajackoyah': 'Prof. Wajackoyah'}})

                words_presidents = words_clean.drop('bigram', axis = 1)

                words_president = words_presidents.groupby('term',  as_index=False, sort=False).agg({'count': 'sum'})
                words_president.drop(words_president.loc[words_president['term']== 'None'].index, inplace=True)
                words_president['count'] = ((words_president['count']/ words_president['count'].sum()) * 100).round(2)

                # Names and counts
                names = words_president['term'].to_list()
                counts = words_president['count'].to_list()

                source = pd.DataFrame({
                    'counts': counts,
                    'names': names
                    })
                bar_chart = alt.Chart(source).mark_bar().encode(
                    y='counts',
                    x='names',
                    )
            
                text = bar_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    ).encode(text='counts')
            
                plt = (bar_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)
            if choice3 == "Polarity of sentiments towards various presidential aspirants for Kenyas 2022 general election":
                st.subheader("Polarity of sentiments towards various presidential aspirants for Kenyas 2022 general election")

                st.write("""
                **_The line chart represents the changes in polarity of the electorate as we head toward the 2022
                election. Score of 1 represents very positive, -1 represents very negative, and 0 represents 
                neautral sentiments_**
                """)

                
                the_list = ['wajackoyah', 'raila',  'ruto', 'deputy', 'baba']

                def aspirant_finder(string):
                    term_return = 'None'
                    for term in the_list:
                        if term in string:
                            term_return = term
            
                    return term_return
                
                df2 = df.loc[mask]
                df3 = df2[['tweet_clean', 'time', 'Polarity']]
                df3['aspirant'] = df3['tweet_clean'].apply(aspirant_finder)

                # presidential aspirants
                words_clean =df3.replace({'aspirant' : {'deputy' : 'Dr. Ruto', 'baba' : 'Mr. Odinga', 'ruto' : 'Dr. Ruto', 'raila': 'Mr. Odinga',  'wajackoyah': 'Prof. Wajackoyah'}})
                words_aspirants = words_clean.drop('tweet_clean', axis = 1)

                words_aspirant = words_aspirants.groupby(['time','aspirant'], as_index=False, sort=False).agg({'Polarity': 'mean'})
                words_aspirant.drop(words_aspirant.loc[words_aspirant['aspirant']== 'None'].index, inplace=True)
                date1 = words_aspirant['time'].to_list()
                polaritys = words_aspirant['Polarity'].to_list()
                aspirant = words_aspirant['aspirant'].to_list()

                source = pd.DataFrame({
                    'polaritys': polaritys,
                    'date': date1,
                    'symbol': aspirant
                    })

                line_chart = alt.Chart(source).mark_line().encode(
                    y='polaritys',
                    x='date',
                    color='symbol',
                    strokeDash='symbol',
                    )
            
                text = line_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    )
            
                plt = (line_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)

            
        else:
            st.error('Error: End date must fall after start date.')

if navigation == "Presidential Election Prediction":
    st.write("At indespensable we try to predict the presidential aspirant most likely to win the forthcoming August 9th, 2022 elections. This prediction is made using various sentiments obtained from social media regarding the general election. YOu can observe the changes in favour of the presidential aspirants for different periods leading towards the election")
    st.write("What dates do you want to get popularity of various political figures?")
    
    st.sidebar.subheader("Dates of interest")
    start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)

    if start < time:
        st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')
        
    end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)

    if start < end and start >= time:
        st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))

        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")
            
        #greater than the start date and smaller than the end date
        mask = (df['time'] > start) & (df['time'] <= end)

        df1 = df.loc[mask]

        # Word finder
        the_list = ['wajackoyah', 'raila',  'ruto', 'deputy', 'baba']

        def word_finder(string):
            term_return = 'None'
            for term in the_list:
                if term in string:
                    term_return = term
            return term_return

        # print(words1)

        df1['term'] = df1['tweet_clean'].apply(word_finder)

        # presidential aspirants
        df1 =df1.replace({'term' : { 'deputy' : 'Dr. William Samoei Ruto', 'baba' : 'Raila Amollo Odinga', 'ruto' : 'Dr. William Samoei Ruto', 'raila': 'Raila Amollo Odinga',  'wajackoyah': 'Prof. George Luchiri Wajackoyah'}})

        df1['Expressions'] = np.where(df1['Polarity'] > 0, 'Positive', 'Negative')
        df1.loc[df1.Polarity == 0, 'Expressions'] = 'Neutral'
        total_neutral = len(df1[df1['Polarity']==0])/3
        df1.drop((df1[df1['Polarity']==0]).index, inplace=True)

        def pol_percent(subset,total):
            neg_percent = ((subset.groupby('Expressions').count())['Polarity'][0]/total)*100
            pos_percent = ((subset.groupby('Expressions').count())['Polarity'][1]/total)*100

            return neg_percent,pos_percent

        df_ruto = df1 [df1 ['term'] == 'Dr. William Samoei Ruto']

        df_raila = df1 [df1 ['term'] == 'Raila Amollo Odinga']

        df_wajackoyah = df1 [df1 ['term'] == 'Prof. George Luchiri Wajackoyah']

        records_raila = len(df_raila)
        records_ruto = len(df_ruto)
        records_wajackoyah = len(df_wajackoyah)
        total_records = records_raila + records_ruto + records_wajackoyah + total_neutral

        ruto_total_percent = pol_percent(df_ruto,records_ruto)
        raila_total_percent = pol_percent(df_raila,records_raila)
        wajackoyah_total_percent = pol_percent(df_wajackoyah,records_wajackoyah)
        undecided_total_percent = total_neutral/total_records * 100

        ruto_pos = (ruto_total_percent[1] + (raila_total_percent[0] + wajackoyah_total_percent[0])/2) * (records_ruto/total_records )
        raila_pos =(raila_total_percent[1] + (ruto_total_percent[0] + wajackoyah_total_percent[0])/2) * (records_raila/total_records )
        wajackoyah_pos = (wajackoyah_total_percent[1] + (ruto_total_percent[0] + raila_total_percent[0])/2) * (records_wajackoyah/total_records)
        undecided_pos_percent = total_neutral/total_records * 100


        counts = [ruto_pos, raila_pos, wajackoyah_pos, undecided_pos_percent]
        names =  ['ruto\'s Favour' ,'raila\'s Favour','wajackoyah\'s Favour', 'Undecided Voters']
        source = pd.DataFrame({
            'counts': counts,
            'names': names
            })
        source1 = source.sort_values(by='counts', ascending = True)
        bar_chart = alt.Chart(source1).mark_bar().encode(
            y='counts',
            x='names',
            )
        
        text = bar_chart.mark_text(
            align='left',
            baseline='middle',
            dx=3  # Nudges text to right so it doesn't appear on top of the bar
            ).encode(text='counts')
        
        plt = (bar_chart + text).properties(height=600)
        st.altair_chart(plt, use_container_width=True)






    





        
    








