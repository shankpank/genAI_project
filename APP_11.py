import streamlit as st
from streamlit_chat import message
import utils3
import pyperclip
import pandas as pd
 
def initialize_session_state():
    """
    Session State is a way to share variables between reruns, for each user session.
    """
 
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to questions fetched from Database."])
    st.session_state.setdefault('past', ["Hello Buddy!"])
 
def display_chat():
    reply_container = st.container()
    container = st.container()
 
    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me questions from MMS3", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
    #     st.markdown(utils3.convert_to_excel(response), unsafe_allow_html=True)
    #     st.success('Downloaded the excel!!!')
       
        #Check if user submit question with user input and generate response of the question
        if submit_button and user_input:
            generate_response(user_input)
   
    #Display generated response to streamlit web UI
    display_generated_responses(reply_container)
 
def generate_response(user_input):
    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input)
 
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
 
def conversation_chat(user_input):
    response = utils3.get_response(user_input)
    print(response)
    
    # if st.button('Copy📋', key='copy_button'):
    #     pyperclip.copy(response)
    #     st.success('Text copied successfully!!!')

    st.markdown(utils3.convert_to_excel(response), unsafe_allow_html=True)
 
    # if st.button('Download Excel⬇️'):
    #     st.markdown(utils3.convert_to_excel(response), unsafe_allow_html=True)
    #     st.success('Downloaded the excel!!!')
    #     #( working.....)
       
    return response
 
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
   
    display_chat()
 
if __name__ == "__main__":
    main()
