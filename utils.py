import os
from dotenv import load_dotenv
load_dotenv()
import ast
import re
import sqlite3
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from typing import List, Optional
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, chain
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
import pandas as pd
import base64
 
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
azure_openai_gpt_deployment = os.getenv('AZURE_OPENAI_GPT_DEPLOYMENT')
azure_openai_embedding_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
 
database_uri = os.getenv('DATABASE_URI')
talent_data_path = os.getenv('TALENT_DATA_PATH')
film_data_path = os.getenv('FILM_DATA_PATH')
 
 
llm = AzureChatOpenAI(
  azure_endpoint = azure_openai_endpoint,
  api_key = azure_openai_key,  
  api_version = azure_openai_api_version,
  azure_deployment=azure_openai_gpt_deployment
)
 
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_openai_embedding_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key
)
 
class Table(BaseModel):
    """Table in SQL database."""
 
    name: str = Field(description="Name of table in SQL database.")
 
class Search(BaseModel):
    """Search for film names & talent names"""
 
    query: str = Field(description="Query to look up")
    query_type: str = Field(description="Which way to go. Should be 'FILM' or 'TALENT'")
    print()
 
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return res
 
def get_db_connection() -> SQLDatabase:
    return SQLDatabase.from_uri(database_uri)
 
def create_index(db) -> dict:
    talent_name_data = query_as_list(db, "SELECT distinct TALENT_FULL_NAME from talent_info")
    film_name_data = query_as_list(db, "SELECT distinct TITLE_NAME from title_info")
    talent_vectorstore = Chroma.from_texts(talent_name_data, embeddings, collection_name="talent_name", persist_directory= talent_data_path)
    talent_vectorstore.persist()
    film_vectorstore = Chroma.from_texts(film_name_data, embeddings, collection_name="film_name", persist_directory= film_data_path)
    film_vectorstore.persist()
    print("Number of Artists : ",len(talent_name_data))
    print("Number of Movies : ",len(film_name_data))
    return talent_vectorstore, film_vectorstore
 
def create_table_chain(db):
    table_names = "\n".join(db.get_usable_table_names())
    system = """Return the names of the SQL tables that are relevant to the user question. \
    The tables are:
 
    Film
    Business"""
 
    category_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)
 
    def get_tables(categories: List[Table]) -> List[str]:
        tables = []
        for category in categories:
            if category.name == "Film":
                tables.extend(
                    [
                        "talent_info",
                        "title_boxoffice_info",
                        "title_info"
                    ]
                )
            elif category.name == "Business":
                tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
        return tables
   
    def get_tables_details(tables: List[str]) -> List[str]:
        return db.get_table_info(tables)
 
    table_chain = category_chain | get_tables | get_tables_details
    return {"input": itemgetter("question")} | table_chain
 
def create_retriever_chain(talent_vectorstore, film_vectorstore):
    system = """You are an AI Assistant that helps in segregating the user queries asked upton database based on type of the question\
    You need to segregate the user query either in 'FILM' type or 'TALENT' type. If the question asked by user relates any \
    info about movie return the keyword 'FILM' and if the query refers to any person return 'TALENT'.\n\
    ***You must be returning any one of the keyword 'FILM' or 'TALENT'***"""
 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}")
        ]
    )
 
    output_parser = StrOutputParser()
 
    query_analyser = {"question": RunnablePassthrough()} | prompt | llm | output_parser
 
    talent_retriever = talent_vectorstore.as_retriever(search_kargs = {"k": 1})
    film_retriever = film_vectorstore.as_retriever(search_kargs = {"k": 1})
    retrievers = {
        "FILM": film_retriever,
        "TALENT": talent_retriever
    }
 
    @chain
    def custom_chain(question):
        response = query_analyser.invoke(question)
        print(response)
        if 'FILM' in response:
            response = 'FILM'
        elif 'TALENT' in response:
            response = 'TALENT'
        retriever = retrievers[response]
        print(retriever)
        result = retriever.invoke(question)
        print(result)
        return result
   
   
    return (itemgetter("question") | custom_chain | (lambda docs: "\n".join(doc.page_content for doc in docs)))
 
def create_query_chain(db):
    
    system = """You are a SQLite expert. Given an input question, create a syntactically \
    correct SQLite query to run. Unless otherwise specificed, do not return more than \
    {top_k} rows. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.\
    Pay attention to use date('now') function to get the current date, if the question involves "today".\n**Only return the SQL query**\n\n\
    Here is the relevant table info: {table_info}\nOptimize this query to have the best results.\nHere is a non-exhaustive \
    list of possible feature values. If filtering on a feature value make sure to check its spelling \
    against this list first:\n\n{proper_nouns}"""
 
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])
 
    query_chain = create_sql_query_chain(llm, db, prompt=prompt)
 
    return query_chain
 
def parser_output():
    output_parser = StrOutputParser()
    chain = llm | output_parser
    return chain
 
def search(question: str) -> dict:
    db = get_db_connection()
   
    try:
        print('Using Existing Chromadb')
        talent_vectorstore = Chroma(collection_name="talent_name", persist_directory=talent_data_path, embedding_function=embeddings)
        film_vectorstore = Chroma(collection_name="film_name", persist_directory=film_data_path, embedding_function=embeddings)
    except:
        print('Creating new Chromadb')
        talent_vectorstore, film_vectorstore = create_index(db)
   
    table_chain = create_table_chain(db)
    retriever_chain = create_retriever_chain(talent_vectorstore, film_vectorstore)
    query_chain = create_query_chain(db)
    combined_chain = (RunnableParallel(
        {
            "question": itemgetter("question"),
            "table_info": table_chain,
            "proper_nouns": retriever_chain,
        }
    )
    | query_chain)
   
    sql_query = combined_chain.invoke({"question": question})
    print("SQL Query",sql_query)
    query_result = db.run(sql_query)
    print("Query Result",query_result)
    #return {"sql_query": sql_query, "query_result": query_result}
    return query_result
 
 
def convert_to_excel(output):
    prompt_template = f"You an AI Assistant that converts a chatbot's response into pipe delimited data.Based upon following instructions generate the response for given context:\
    **Instructions**\
    --Only return the tabular data.\
    --The first row of table must be the column header immediately followed by tabular data.\
    --There shouldn't be any row seperator between header and data.\
    --Do not name the coulmns as Column1 etc. Give proper naming defining the data column.\
    --Do not include rows such as Column1|Column2|Column3.... and -|-|-....\
    --If the context dosen't provide any valid column names for data, makeup appropriate column names best fitting the data.\
    Refer the below context to generate the response.\
    Context:\n{output}"
    data = llm.invoke(prompt_template).content
    print(data)
    df = pd.DataFrame([x.split('|') for x in data.split('\n')])
    df=df.to_csv(index=False,header=False)
    b64 = base64.b64encode(df.encode()).decode()
    href = f'<button type="button"><a href="data:file/csv;base64,{b64}" download="generated_response.csv">Download Response⬇️</a></button>'
    return href

 
def get_response(user_input):
    output_parser = StrOutputParser()
    chain = llm | output_parser
 
    try:
        response = search(user_input)
 
        if response:
            response = chain.invoke(f"Based upon following instructions generate the response for given context:\
                                    **Instructions**\
                                    --Your task is to generate a human readable response for given context from database.\
                                    --The context includes the User's question and answer obtained from db.\
                                    --If the answer is a singular value answer it in a interactice manner.\
                                    --If the answer is too long or contains multiple rows and columns display it in the *tabular format*.\
                                    --If the answer includes any Numerical values display them without decimal places e.g. 20, 1933, 3456 etc.\
                                    --Always include '$' for any monetory values and make necessary formatting to make them presentable.\
                                    **Context**\
                                    -User's Question: {user_input}\
                                    --Database result:\
                                    {response}")
            
       
        else:
            response = chain.invoke(f"Generate a concise response for below context:\nApologies, there is no relevant information about your question \'{user_input}\'.\Try asking a differnt question or rephrase the question")
 
    except Exception as e:
        response = chain.invoke(f"Generate a concise response for below context:\n \
                                 Apologies, there is no relevant information about your question \'{user_input}\'.\Try asking a differnt question or rephrase the question")
        #raise e
   
    return response
