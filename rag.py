####---------------------------------------------RAG Pipeline-----------------------------------------------####

import streamlit as st
import os
import sys
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser, Document
from langchain_core.documents import Document

from src.config.appconfig import GROQ_API_KEY, GROQ_MODEL_NAME
from src.tools.edu_neck import EDU_SYS_PROMPT_NECK
# from src.tools.edu_head import EDU_SYS_PROMPT_HEAD
# from src.tools.edu_prompt_combo import SYS_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAG:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 100):
        self.persist_directory = r"src/docs/chroma"
        self.data_path = r"src/data/survey_data.csv"
        self.llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.1)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.groq_api_key = GROQ_API_KEY
        self.model_name = GROQ_MODEL_NAME
        self.chunk_size = chunk_size    
        self.chunk_overlap = chunk_overlap
        self.initialize_session_state()
        self.embeddings = self.load_embeddings()
        self.vector_store = None
        self.loaded_doc = None
        self.logger = self.setup_logger()
        
    
    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler('./log/rag_pipeline.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        return logger
    
    #-----------------------------------------------Initialize Session State---------------------------------#
    @staticmethod
    
    def initialize_session_state():
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    #-----------------------------------------------Load Documents---------------------------------#
    def document_loader(self):
        try:
            self.logger.info("Loading and preprocessing document")
            csv_loader = CSVLoader(file_path=self.data_path, encoding='utf-8')
            self.loaded_doc = csv_loader.load()
            self.logger.info(f"Successfully loaded CSV file data")
        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            raise
    
    #-----------------------------------------------Load Embeddings---------------------------------#    
    @st.cache_resource    
    
    def load_embeddings(_self):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            # _self.logger.info("Embeddings loaded successfully")
            return embeddings
        except Exception as e:
            # _self.logger.error(f"Error loading embeddings: {str(e)}")
            st.error(f"Error loading embeddings: {str(e)}")
            return None

    #-----------------------------------------------Creating Vector Store---------------------------------#       
    @st.cache_resource
    
    def create_vector_store(_self, _documents: List[Document]) -> Optional[Chroma]:
        if not _documents:
            _self.logger.warning("No documents provided to the vector store.")
            return None

        try:
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_self.chunk_size,
            chunk_overlap=_self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(_documents)
            
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=_self.load_embeddings(),
                persist_directory=_self.persist_directory
            )
            vector_store.persist()
            
            _self.logger.info(f"Vector store created with {len(texts)} chunks.")
            return vector_store
        
        except Exception as e:
            _self.logger.error(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating vector store: {str(e)}")
            return None
    
    #-----------------------------------------------Creating Retriever---------------------------------# 
    def retriever(self, user_query: str) -> Dict[str, Any]:
        self.logger.info('Retrieving Relevant Documents')
        if not os.path.exists(self.persist_directory):
            self.document_loader()
            self.vector_store = self.create_vector_store(self.loaded_doc)
        
        if not self.vector_store:
            self.logger.error("Vector store not initialized.")
            return {"error": "Vector store not initialized"}
        
        try:
            
            k = 4
            relevant_docs = self.vector_store.similarity_search(user_query, k=k)
            self.logger.info(f"Retrieved {len(relevant_docs)} relevant documents.")
            
            # retriever = self.vector_store.as_retriever(
            #     search_type="similarity",
            #     search_kwargs={"k": k}  
            # )
            
            return {
                "relevant_content": [doc.page_content for doc in relevant_docs],
                "metadata": [doc.metadata for doc in relevant_docs]
            }
            
        except Exception as e:
            self.logger.error(f"Error in retriever: {str(e)}")
            return {"error": str(e)}
    
    
    def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
    
    #-----------------------------------------------Creating RAG Pipeline---------------------------------# 
    def create_rag_pipeline(self):
        if not self.vector_store:
            self.logger.error("Vector store not initialized. Cannot create RAG application.")
            return None
        
        k = 4
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        # prompt = PromptTemplate(
        #     input_variables=['context', 'input'], 
        #     template=EDU_SYS_PROMPT_NECK
        # )
        
        prompt = PromptTemplate.from_template(EDU_SYS_PROMPT_NECK)
        
        retriever_chain = (
            {"context": retriever | 
                                    self.format_docs, 
                                    "chat_history": RunnablePassthrough(), 
                                    "question": RunnablePassthrough()} 
            | prompt 
            | self.llm
            | StrOutputParser()
        )
        
        return retriever_chain
    
    
    #-----------------------------------------------Chat with DOC---------------------------------# 
    def chat_with_doc(self, user_query: str) -> Dict[str, str]:
        try:
            rag_chain = self.create_rag_pipeline()
            if not rag_chain:
                raise ValueError("RAG pipeline could not be created.")
            
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = rag_chain.invoke({"question": user_query, "chat_history": chat_history})
            
            self.memory.save_context({"input": user_query}, {"output": response})
            st.session_state.chat_history.append(('Human', user_query))
            st.session_state.chat_history.append(('AI', response))
            
            self.logger.info("Successfully generated response.")
            return {'result': response}
        except Exception as e:
            error_message = f"Error in chat_with_doc: {str(e)}"
            self.logger.error(error_message)
            return {'error': error_message}
            

    def get_chat_history(self) -> List[tuple]:
        return st.session_state.chat_history
    

    def clear_chat_history(self):
        st.session_state.chat_history = []
        st.session_state.messages = []
        self.memory.clear()
        self.logger.info("Chat history cleared.")
        st.success("Conversation history cleared successfully!")