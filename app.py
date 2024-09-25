import streamlit as st
import streamlit.components.v1 as components
import json
import os
from typing import List, Any
import re
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv;load_dotenv()
import warnings;warnings.filterwarnings("ignore")

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.exceptions import LangChainException

#-----------------------------------Set up Logging-------------------------------
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#-----------------------------------Prompts-------------------------------
from src.tools.edu_head import EDU_SYS_PROMPT_HEAD
from src.tools.eduai_query_gen import get_columns_descriptions


from rag import RAG
#---------------------------------------------Create a Streamlit app-----------------------------------------------
# st.set_page_config(
#     page_title="EduAI",
#     page_icon="📚",
#     layout="centered",
#     # initial_sidebar_state="collapsed"
# )

# Define a color scheme
PRIMARY_COLOR = "#4A90E2"
SECONDARY_COLOR = "#F5A623"
BACKGROUND_COLOR = "#F0F4F8"
TEXT_COLOR = "#333333"

# Custom CSS
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {{
        font-family: 'Roboto', sans-serif;
        # background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    
    .stApp {{
        max-width: 1200px;
        margin: 0 auto;
    }}

    h1, h2, h3 {{
        color: {PRIMARY_COLOR};
    }}

    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}

    .stTextInput>div>div>input {{
        border-radius: 5px;
        border: 1px solid #E0E0E0;
    }}

    .stFileUploader>div {{
        border-radius: 5px;
        border: 2px dashed {PRIMARY_COLOR};
        padding: 2rem;
    }}

    .stFileUploader>div:hover {{
        border-color: {SECONDARY_COLOR};
    }}

    .css-145kmo2 {{
        border: none;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .css-1d391kg {{
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.title("📚 EduAI - Your Educational Assistant")
# Sidebar for app controls and information
with st.sidebar:
    st.header("📁 EduAi Settings")
    st.markdown("Customize your EduAI experience here.")
    
    st.markdown("---")
    st.markdown("### About EduAI")
    st.markdown("""
    EduAI is your intelligent educational assistant, powered by advanced AI and RAG technology. 
    It's designed to help students, educators, and university stakeholders analyze educational data 
    and provide valuable insights.
    """)

# How to Use
# st.sidebar.header("🚀 How to Use")
# st.sidebar.markdown("""
# 1. **Upload your PDF**: Use the sidebar to upload your PDF document.
# 2. **Start chatting**: Once your document is uploaded, use the chat interface to ask questions.
# 3. **Explore insights**: The AI will analyze your document and provide relevant answers.
# """)


#-------------------------------------Pydantic model for CSV Chat--------------------------------
class CSVChat(BaseModel):
    """
    Input schema for analyzing CSV file.
    """
    query: str = Field(..., description='Query about the CSV file')

#-----------------------------------------------Input validation---------------------------------
def validate_user_input(user_input: str) -> bool:
    if not user_input or not re.match(r'^[a-zA-Z0-9\s\.\,\?\!]+$', user_input):
        st.warning("Please enter a valid query using alphanumeric characters and basic punctuation.")
        return False
    return True
   
#----------------------------------------------------Function to process chat
def process_chat(model, user_input: str, chat_history: List[Any]) -> Any:
    return model.invoke({
        'chat_history': chat_history,
        'input': user_input,
        'get_columns_descriptions': get_columns_descriptions()
    })

class CSVFileHandler(FileSystemEventHandler):
    def __init__(self, rag_instance):
        self.rag = rag_instance

    def on_modified(self, event):
        if event.src_path == self.rag.data_path:
            logger.info(f"CSV file {event.src_path} has been modified. Updating vector store.")
            self.rag.document_loader()
            self.rag.vector_store = self.rag.create_vector_store(self.rag.loaded_doc)
            st.session_state['vector_store_updated'] = True

@st.cache_resource
def setup_file_watcher(_rag_instance):
    event_handler = CSVFileHandler(_rag_instance)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(_rag_instance.data_path), recursive=False)
    observer.start()
    return observer


#------------------------------------------------MAIN APPLICATION-----------------------------------------------------
def main():
    #------------------------------------------------------Initialize RAG-----------------------------------------
    rag = RAG()
    rag.setup_logger()
    
    # Ensure the document loader is called at least once to initialize the vector store
    if not rag.vector_store:
        logger.info("Vector store is not initialized. Loading documents and creating vector store.")
        rag.document_loader()  # Load the CSV data
        rag.vector_store = rag.create_vector_store(rag.loaded_doc)  # Create vector store
        if not rag.vector_store:
            rag.logger.error("Vector store creation failed.")
            rag.logger.error("Vector store failed to initialize. Please check the CSV file or data.")
        else:
            rag.logger.info("Vector store initialized successfully.")
    
    # Set up file watcher
    observer = setup_file_watcher(rag)
    
    #---------------------------------------CONVERTING TO FUNCTION DECLARATION OBJECT---------------------------------
    pdf_chats_func = convert_to_openai_function(CSVChat)

    #---------------------------------------------Create Chat Template----------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(EDU_SYS_PROMPT_HEAD),
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate.from_template('{input}')
        ]
    )

    
    #---------------------------------------------Initialize Chat Models--------------------------------------
    chat_model = rag.llm
    chat_with_tools = chat_model.bind_tools(tools=[pdf_chats_func]) 
    chain =  prompt | chat_with_tools

    #---------------------------------------------Initialize Session State--------------------------------
    rag.initialize_session_state()

    #-----------------------------------------Display Chat History--------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    #------------------------------------------User input
    if user_input := st.chat_input('Message EduAI 💬', key='user_input'):
        if user_input and validate_user_input(user_input):
            st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
    
        with st.chat_message('assistant', avatar="🤖"):
            with st.spinner('Thinking...'):
                try:
                    
                    # # Check if vector store is still valid before proceeding with the chat
                    # if not rag.vector_store:
                    #     logger.error("Vector store is missing during chat processing.")
                    #     st.error("Vector store not available. Please upload or update the CSV file.")
                    #     return
                    
                    response = process_chat(chain, user_input ,st.session_state.chat_history)
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(response)
                    
                    logger.info(f"Initial AI response: {response}")
                
                    if response.content:
                        st.markdown(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                    
                    # print(f"Response: {response}, '\n")
                    # print(f"Response Additional Kwargs: {response.additional_kwargs}, '\n")
                    
                    elif response.additional_kwargs.get('tool_calls'):
                        # Extracting information
                        tool_calls = response.additional_kwargs.get('tool_calls', [])
                        for call in tool_calls:
                            #  EXTRACT THE PARAMETERS TO BE PASSED TO THE FUNCTIONS
                            function = call.get('function', {})
                            function_name = function.get('name')
                            function_args = json.loads(function.get('arguments', '{}'))

                            # PERFORMS THE FUNCTION CALL OUTSIDE THE LLM MODEL
                            if function_name == 'CSVChat':
                                with st.status('Analyzing CSV File...', expanded=True) as status:
                                    api_response = rag.retriever(function_args.get('query', ''))
                                    status.update(label='Analysis Complete', state='complete', expanded=False)
                                
                                # logger.info(f"API response: {api_response}")
                                
                                # PARSE THE RESPONSE OF THE API CALLS BACK INTO THE MODEL
                                tool_message = ToolMessage(content=str(api_response), name=function_name, tool_call_id=call.get('id'))
                                ai_response = process_chat(chain, str(tool_message), st.session_state.chat_history)
                                
                                logger.info(f"Final AI response: {ai_response}")

                            if ai_response.content:
                                # APPEND THE FUNCTION RESPONSE AND AI RESPONSE TO BOTH THE CHAT_HISTORY AND MESSAGE HISTORY FOR STREAMLIT 
                                st.markdown(ai_response.content)
                                st.session_state.chat_history.extend([tool_message, ai_response])
                                st.session_state.messages.append({"role": "assistant", "content": ai_response.content})
                            else:
                                st.warning("The AI didn't provide a clear response. Here's what I found in the document:")
                                st.json(api_response)
                                st.error("The AI didn't provide a response. Please try again.")

                    else:
                        st.warning("The AI couldn't generate a response based on the document. Please try rephrasing your question.")
                
                    st.rerun()
                    
                
                
                except (Exception,LangChainException) as e:
                    logger.error(f"An error occurred: {e}")
                    st.error("An unexpected error occurred. Please try again later.")
    
    
    # Check if vector store has been updated
    if st.session_state.get('vector_store_updated', False):
        st.success("The CSV file has been updated. The vector store has been refreshed with the latest data.")
        st.session_state['vector_store_updated'] = False
    
    #------------------------------------------Clear Conversation------------------------------------------
    if st.sidebar.button('Clear Conversation'):
        rag.clear_chat_history()
        st.rerun()
        

if __name__ == "__main__":
    main()

# #---------------------------------------------------------FOOTER------------------------------------------------

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        text-align: center;
    }
    .footer a {
        color: #007bff;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer content
st.markdown('<div class="footer">Developed with ❤️ by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a></div>', unsafe_allow_html=True)
