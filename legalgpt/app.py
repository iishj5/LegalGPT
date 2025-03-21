# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import os
# import time

# # Streamlit configuration
# st.set_page_config(page_title="LegalCoder", layout="centered")
# st.image("D:/Projects/LegalGPT-main/legalgpt/images/banner.png", use_column_width=True)

# # Model configuration
# MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
# LOCAL_MODEL_PATH = "./legal_model"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# @st.cache_resource
# def load_model():
#     """Load or download and cache the model locally"""
#     try:
#         # Try loading local model first
#         tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True).to(DEVICE)
#     except:
#         # Download and save model if local copy not found
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        
#         # Save model locally
#         os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
#         tokenizer.save_pretrained(LOCAL_MODEL_PATH)
#         model.save_pretrained(LOCAL_MODEL_PATH)
    
#     return tokenizer, model

# # Load model and tokenizer
# tokenizer, model = load_model()

# # Legal prompt template
# LEGAL_PROMPT = """[INST] <<SYS>>
# You are a legal assistant for Indian law. Answer questions about IPC with:
# 1. Clear bullet points
# 2. Relevant sections
# 3. Simple explanations
# 4. Practical examples
# <</SYS>>

# Question: {question}
# Answer: [/INST]"""

# def generate_legal_response(question):
#     """Generate legal response using DeepSeek model"""
#     full_prompt = LEGAL_PROMPT.format(question=question)
    
#     inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    
#     try:
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.1
#         )
#         return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# # Chat interface
# st.title("Legal Query Assistant")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# if prompt := st.chat_input("Ask your legal question..."):
#     # Display user message
#     st.chat_message("user").write(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Generate response
#     with st.chat_message("assistant"):
#         with st.spinner("Analyzing IPC provisions..."):
#             response = generate_legal_response(prompt)
            
#             # Stream the response
#             response_container = st.empty()
#             full_response = ""
#             for chunk in response.split():
#                 full_response += chunk + " "
#                 response_container.markdown(full_response + "▌")
#                 time.sleep(0.05)
#             response_container.markdown(full_response)
    
#     st.session_state.messages.append({"role": "assistant", "content": response})
import time
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

from footer import footer

# Set the Streamlit page configuration and theme
st.set_page_config(page_title="LegalGPT", layout="centered")

# Display the logo image
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("D:/Projects/LegalGPT-main/legalgpt/images/banner.png", use_column_width=True)

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

embeddings = load_embeddings()
db = FAISS.load_local("D:/Projects/LegalGPT-main/legalgpt/faiss_index", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law, ensuring it reflects general application]
- [Provide a concise explanation of how the law is typically interpreted or applied]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information that directly relates to the user's query]
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

api_key = "AIzaSyAo1MwoUb0zvVLLBGZAj8Jy2SGkKbF561M"
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.5, max_tokens=1024, google_api_key=api_key)

qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=st.session_state.memory, retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})

def extract_answer(full_response):
    """Extracts the answer from the LLM's full response by removing the instructional text."""
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        answer_end = len(full_response)
        return full_response[answer_start:answer_end].strip()
    return full_response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

input_prompt = st.chat_input("Say something...")
if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"*You:* {input_prompt}")

    st.session_state.messages.append({"role": "user", "content": input_prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            answer = extract_answer(result["answer"])

            # Initialize the response message
            full_response = "⚠ *Gentle reminder: We generally ensure precise information, but do double-check.* \n\n\n"
            for chunk in answer:
                # Simulate typing by appending chunks of the response over time
                full_response += chunk
                time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button('🗑 Reset All Chat', on_click=reset_conversation):
            st.experimental_rerun()

# Define the CSS to style the footer
footer()