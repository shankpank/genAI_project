import streamlit as st
from streamlit_chat import message
from langchain_core.output_parsers import StrOutputParser
import utils
 
def initialize_session_state():
    """
    Session State is a way to share variables between reruns, for each user session.
    """
 
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to questions fetched from Database."])
    st.session_state.setdefault('past', ["Hello Buddy!"])
 
def display_chat(chain):
    reply_container = st.container()
    container = st.container()
 
    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me questions from MMS3", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
       
        #Check if user submit question with user input and generate response of the question
        if submit_button and user_input:
            generate_response(user_input,chain)
   
    #Display generated response to streamlit web UI
    display_generated_responses(reply_container)
 
def generate_response(user_input,chain):
    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input,chain)
 
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
 
def conversation_chat(user_input,chain):
    try:
        response = utils.search(user_input)
        print(response)
        final_response = chain.invoke(f"Based on the following information generate human readable response: (**Use only the below info to generate response): \
                                      Question: {user_input}\
                                      Answer: {response}")
        return final_response
    except Exception as e:
        error_statement = f"Sorry there is no relevant information about your question \'{user_input}\'.\
                            Try asking a differnt question or rephrase the question :("
        return error_statement
   
 
def display_generated_responses(reply_container):
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
 
def main():
    initialize_session_state()
   
    st.title("MMS3 Chatbot🤖")
   
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
 
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    chain=utils.parser_output()
    display_chat(chain)
 
if __name__ == "__main__":
    main()
 