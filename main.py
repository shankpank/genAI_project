from dotenv import load_dotenv
load_dotenv() ## load all the environemnt variables
# import sqlparse
import streamlit as st
import os
import pandas as pd
import sqlite3
from langchain.sql_database import SQLDatabase
 
import google.generativeai as genai
## Configure Genai Key
 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
title_df = pd.read_excel(r"C:\Data\Development\GenAI\GenAI_Usecase\WBGenAIRequirement\Source Documents\Title_Info.xlsx")
talent_df = pd.read_excel(r"C:\Data\Development\GenAI\GenAI_Usecase\WBGenAIRequirement\Source Documents\Talent_Info.xlsx")
title_boxoffice_df = pd.read_excel(r"C:\Data\Development\GenAI\GenAI_Usecase\WBGenAIRequirement\Source Documents\Title_BoxOffice_Info.xlsx")
 
conn = sqlite3.connect(r"C:\Data\Development\GenAI\GenAI_Usecase\WBGenAIRequirement\wbd_data.sqlite")
c = conn.cursor()
 
c.execute('CREATE TABLE if not exists title_info (PICTURE_NR INTEGER, RELEASE_NR INTEGER , TITLE_NAME TEXT, PRIMARY_DISTRIBUTOR TEXT, SECONDARY_DISTRIBUTORS TEXT, PRIMARY_GENRE TEXT, SECONDARY_GENRES TEXT, RATING TEXT, RELEASE_DATE TEXT, RELEASE_YEAR INTEGER, RELEASE_FORMAT TEXT, WIDE_RELEASE_DATE TEXT, PRODCO_FULL_NAME TEXT)')
c.execute('CREATE TABLE if not exists talent_info (RELEASE_NR INTEGER, PICTURE_FULL_NAME TEXT, TALENT_FULL_NAME TEXT, JOB_FULL_NAME TEXT)')
c.execute('CREATE TABLE if not exists title_boxoffice_info (RELEASE_NR INTEGER, PICTURE_FULL_NAME TEXT, CUME_GROSS REAL, OPENING_DAY_AMOUNT REAL, OPENING_DAY_LOCS INTEGER, OPENING_WEEKEND_AMOUNT REAL, OPENING_WEEKEND_LOCS INTEGER, WIDE_OPNG_DAY_AMOUNT REAL, WIDE_OPNG_DAY_LOCS INTEGER, WIDE_OPNG_WEEKEND_AMOUNT REAL, WIDE_OPNG_WEEKEND_LOCS INTEGER)')
 
conn.commit()
 
print(title_df.to_sql('title_info', conn, if_exists='replace', index = False))
print(talent_df.to_sql('talent_info', conn, if_exists='replace', index = False))
print(title_boxoffice_df.to_sql('title_boxoffice_info', conn, if_exists='replace', index = False))
 
 
db = SQLDatabase.from_uri("sqlite:///C:\\Data\\Development\\GenAI\\GenAI_Usecase\\WBGenAIRequirement\\wbd_data.sqlite")
 
 
## Function To Load Google Gemini Model and provide queries as response
 
def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text
 
## Fucntion To retrieve query from the database
 
def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    print(rows)
    conn.commit()
    conn.close()
    return rows
#for row in rows:
#print(row)
 
## Define Your Prompt
prompt=[
"""
You are an expert in converting English questions to SQL query!

CREATE TABLE if not exists title_info (
    PICTURE_NR INTEGER,                     --This refers to the unique id of a picture.
    RELEASE_NR INTEGER PRIMARY KEY,         --Integer field indicating the release number of the picture.
    TITLE_NAME TEXT,                        --This refers to the movie name.
    PRIMARY_DISTRIBUTOR TEXT,               --This refers to the production house in which movie has been made.
    SECONDARY_DISTRIBUTORS TEXT,            --Text field listing any secondary distributors associated with the movie..
    PRIMARY_GENRE TEXT,                     --Text field indicating the primary genre of the title.
    SECONDARY_GENRES TEXT,                  --Text field listing any secondary genres of the title
    RATING TEXT,                            --Text field representing the rating assigned to the title
    RELEASE_DATE TEXT,                      --Text field indicating the release date of the title.
    RELEASE_YEAR INTEGER,                   --Integer field specifying the release year of the title.
    RELEASE_FORMAT TEXT,                    --Text field describing the format of the release (For example IMAX, 3D).
    WIDE_RELEASE_DATE TEXT,                 --Text field indicating the wide release date of the title.
    PRODCO_FULL_NAME TEXT                   --Text field representing the full name of the production company associated with the title.
);

CREATE title_boxoffice_info(
    RELEASE_NR INTEGER PRIMARY KEY,         --Integer field indicating the release number of the picture.
    PICTURE_FULL_NAME TEXT,                 --Full name of the picture or movie.
    CUME_GROSS REAL,                        --Cumulative gross amount collected by movie.
    OPENING_DAY_AMOUNT REAL,                --Amount earned on the opening day.
    OPENING_DAY_LOCS INTEGER,               --Number of locations where the picture was released on its opening day.
    OPENING_WEEKEND_AMOUNT REAL,            --Amount earned on its opening weekend.
    OPENING_WEEKEND_LOCS INTEGER,           --Number of locations where the picture was released on its opening weekend.
    WIDE_OPNG_DAY_AMOUNT REAL,              --Amount earned on the wide opening day.
    WIDE_OPNG_DAY_LOCS INTEGER,             --Number of locations where the picture was released on the wide opening day.
    WIDE_OPNG_WEEKEND_AMOUNT REAL,          --Amount earned on the wide opening weekend.
    WIDE_OPNG_WEEKEND_LOCS INTEGER          --Number of locations where the picture was released on the wide opening weekend.
);

CREATE TABLE talent_info (
    RELEASE_NR INTEGER PRIMARY KEY,         --Integer field indicating the release number of the picture.
    PICTURE_FULL_NAME TEXT,                 --Full name of the picture or movie.
    TALENT_FULL_NAME TEXT,                  --This refers to cast and crew of the movie.
    JOB_FULL_NAME TEXT                      --This refers to the job the cast and crew did for the picture.
);
 
-- RELEASE_NR.title_info can be joined with RELEASE_NR.title_boxoffice_info
-- RELEASE_NR.title_info can be joined with RELEASE_NR.talent_info

-- If the user enters any incorrect input and its realated to database, make some autocorrections and retrieve desired results.
-- If the user asks a question which isn't related to database, Simply respond 'I Couldn't answer your question.'
"""
]
 
## Streamlit App
 
st.set_page_config(page_title="I can Retrieve Any SQL query")
st.header("Gemini App To Retrieve SQL Data")
 
question=st.text_input("Input: ",key="input")
 
tab_titles=[
    "Results",
    "Query"
]
submit=st.button("Ask the question")
tabs = st.tabs(tab_titles)
 
 
# if submit is clicked
if submit:
    response=get_gemini_response(question,prompt)
    print('Query is: ',response)
    print(type(response))

    with tabs[1]:
        st.write(response)
    #sql_query = response.text
    response=response.replace("```sql","")
    response=response.replace("```","")
    #response = sqlparse.format(response.split("```sql")[-1].strip(), reindent=True)
    
    #print(len(response))
    response=read_sql_query(response.strip(),"wbd_data.sqlite")
    print('Result is: ',response)
    st.subheader("The Response is:-")
    for row in response:
        print(row)
        with tabs[0]:
            st.write(row)
        #st.header(row)