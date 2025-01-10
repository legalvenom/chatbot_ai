import streamlit as st
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

class ChatbotRAG:
    def __init__(self, csv_path):
        """
        Initialize the RAG-based chatbot
        Args:
            csv_path (str): Path to the CSV file containing conversation data
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.load_data(csv_path)
        self.setup_vectorstore()
        self.setup_chain()
        
    def load_data(self, csv_path):
        """
        Load and preprocess the conversation data
        """
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        # Combine prompt and response for context
        self.conversations = []
        for _, row in df.iterrows():
            context = f"Question: {row['prompt']}\nAnswer: {row['response']}"
            self.conversations.append(context)
            
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.text_chunks = text_splitter.create_documents(self.conversations)
        
    def setup_vectorstore(self):
        """
        Set up the vector store using FAISS and HuggingFace embeddings
        """
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=self.text_chunks,
            embedding=self.embeddings
        )
        
    def setup_chain(self):
        """
        Set up the conversational chain using the newer LangChain Expression Language (LCEL)
        """
        # Initialize language model
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            task="text2text-generation",
            model_kwargs={
                "max_new_tokens": 250,
                "temperature": 0.7  # Moved temperature into model_kwargs
            },
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        # Create prompt template
        template = """Answer the following question based on the provided context:

        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Set up retrieval chain using LCEL
        self.retriever = self.vectorstore.as_retriever()
        
        # Build the LCEL chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_response(self, question):
        """
        Get response for user question
        Args:
            question (str): User's question
        Returns:
            str: Chatbot's response
        """
        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Initialize Streamlit state
def initialize_session_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    
    # Initialize session state
    initialize_session_state()
    
    # Streamlit UI
    st.title("RAG-based Chatbot")
    
    # Load chatbot with local dataset
    if st.session_state.chatbot is None:
        try:
            with st.spinner('Loading chatbot...'):
                csv_path = "dataset.csv"
                st.session_state.chatbot = ChatbotRAG(csv_path)
                st.success("Chatbot initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            return
    
    # Chat interface
    if st.session_state.chatbot is not None:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.get_response(prompt)
                    st.write(response)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()