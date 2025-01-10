from langchain_huggingface import HuggingFaceEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import functools
import pdfplumber
import uuid
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting application...")

MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
MAX_PDF_CHARS = 10000

st.set_page_config(page_title="FinBot", page_icon="💰")

@st.cache_resource
def get_llm_hf_inference(model_id="MODEL_ID", max_new_tokens=2000, temperature=0.1):
    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    return HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/{model_id}",
        task="text-generation",
        client=client,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

# Fungsi read_pdf
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def get_pdf_context(pdf_text, option, max_chars=6000):
    if len(pdf_text) <= max_chars:
        return pdf_text
    
    if option == "First part":
        return pdf_text[:max_chars] + f"\n\n... (truncated, total length: {len(pdf_text)} characters)"
    elif option == "Last part":
        return f"... (truncated, total length: {len(pdf_text)} characters)\n\n" + pdf_text[-max_chars:]
    else:  # Distributed sample
        chunk_size = max_chars // 3
        return (pdf_text[:chunk_size] + 
                f"\n\n... (middle part omitted) ...\n\n" + 
                pdf_text[len(pdf_text)//2 - chunk_size//2:len(pdf_text)//2 + chunk_size//2] +
                f"\n\n... (middle part omitted) ...\n\n" + 
                pdf_text[-chunk_size:])

def safe_api_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

# LLM setup
@st.cache_resource
def get_llm_hf_inference(model_id=MODEL_ID, max_new_tokens=512, temperature=0.1):
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    
    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    return HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/{model_id}",
        task="text-generation",
        client=client,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

# Fungsi untuk melakukan analisis multi-agent PDF
def analyze_pdf_multi_agent(pdf_text, llm):
    analyst = FinancialAnalyst(llm)
    manager = ManagerFinancialAnalyst(llm)
    consultant = FinanceConsultant(llm)

    analysis = "".join(analyst.analyze(pdf_text))
    review = "".join(manager.review(analysis))
    recommendation = "".join(consultant.recommend(analysis, review))

    return {
        "analysis": analysis,
        "review": review,
        "recommendation": recommendation
    }

# Agent classes
class FinancialAnalyst:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "As an expert financial analyst, analyze this financial report:\n\n"
            "{report}\n\n"
            "Provide a detailed analysis covering:\n"
            "1. Revenue and profit trends\n"
            "2. Balance sheet health\n"
            "3. Cash flow analysis\n"
            "4. Key financial ratios\n"
            "5. Comparison with industry benchmarks\n"
            "Use specific numbers from the report in your analysis."
        )

    def analyze(self, report):
        chain = self.prompt | self.llm | StrOutputParser()
        return "".join(chain.stream({"report": report}))

class ManagerFinancialAnalyst:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "As a senior financial manager, review this analysis:\n\n"
            "{analysis}\n\n"
            "Provide a critical review addressing:\n"
            "1. Accuracy of the analysis\n"
            "2. Completeness of the assessment\n"
            "3. Potential risks or opportunities overlooked\n"
            "4. Additional insights based on market knowledge\n"
            "5. Recommendations for further analysis if needed"
        )

    def review(self, analysis):
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.stream({"analysis": analysis})

class FinanceConsultant:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "As an experienced finance consultant, consider this analysis and review:\n\n"
            "Analysis:\n{analysis}\n\n"
            "Review:\n{review}\n\n"
            "Provide comprehensive investment recommendations:\n"
            "1. Overall investment stance (bullish, bearish, or neutral)\n"
            "2. Specific investment strategies\n"
            "3. Potential risks and mitigation strategies\n"
            "4. Short-term and long-term investment outlook\n"
            "5. Diversification suggestions\n"
            "Justify your recommendations based on the provided information."
        )

    def recommend(self, analysis, review):
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.stream({"analysis": analysis, "review": review})

# Comprehensive analysis function
@safe_api_call
def comprehensive_analysis(financial_report, llm):
    analyst = FinancialAnalyst(llm)
    manager = ManagerFinancialAnalyst(llm)
    consultant = FinanceConsultant(llm)

    analysis_placeholder = st.empty()
    review_placeholder = st.empty()
    recommendation_placeholder = st.empty()

    with st.spinner("Generating Financial Analysis..."):
        analysis = ""
        for chunk in analyst.analyze(financial_report):
            analysis += chunk
            analysis_placeholder.markdown(analysis)

    with st.spinner("Generating Manager's Review..."):
        review = ""
        for chunk in manager.review(analysis):
            review += chunk
            review_placeholder.markdown(review)

    with st.spinner("Generating Investment Recommendation..."):
        recommendation = ""
        for chunk in consultant.recommend(analysis, review):
            recommendation += chunk
            recommendation_placeholder.markdown(recommendation)

    return {"analysis": analysis, "review": review, "recommendation": recommendation}

# Chatbot
def get_conversational_chain(llm):
    prompt_template = """
    You are a financial assistant with access to comprehensive financial analyses of multiple reports.
    Use the following pieces of information to answer the user's question:
    
    {context}

    Human: {human_input}
    AI Assistant: Based on the financial analyses we have, let me help you with that:
    """

    prompt = PromptTemplate(
        input_variables=["context", "human_input"],
        template=prompt_template
    )

    memory = ConversationBufferMemory(input_key="human_input", memory_key="chat_history")

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    return chain


def display_document_summary(pdf_text):
    st.sidebar.header("Uploaded Documents Summary")
    max_summary_chars = 1000  # Increased for better overview
    summary = pdf_text[:max_summary_chars] + "..." if len(pdf_text) > max_summary_chars else pdf_text
    st.sidebar.text_area("Content Preview", summary, height=300)
    st.sidebar.text(f"Total characters: {len(pdf_text)}")
    st.sidebar.text(f"Number of words: {len(pdf_text.split())}")
    

# Initilize Session state for avatars
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to FinBot! Please enter your financial question below."}
    ]
    
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "pdf_analyses" not in st.session_state:
    st.session_state.pdf_analyses = {}

if "avatars" not in st.session_state:
    st.session_state.avatars = {'user': None, 'assistant': None}
    
# Initialize session state for user text input
if 'user_text' not in st.session_state:
    st.session_state.user_text = None

# Initialize session state for model parameters
if "max_response_length" not in st.session_state:
    st.session_state.max_response_length = 500
    
if "system_message" not in st.session_state:
    st.session_state.system_message = "You are an AI assistant that helps users understand the content of their uploaded PDF. Answer questions based on the PDF content provided."
    
if "starter_message" not in st.session_state:
    st.session_state.starter_message = "Welcome to FinBot! Please enter your financial question below."

    
# Sidebar
with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF Financial Reports", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        llm = get_llm_hf_inference()
        for uploaded_file in uploaded_files:
            file_id = str(uuid.uuid4())
            if file_id not in st.session_state.pdf_analyses:
                pdf_text = read_pdf(uploaded_file)
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    analysis = analyze_pdf_multi_agent(pdf_text, llm)
                    st.session_state.pdf_analyses[file_id] = {
                        "name": uploaded_file.name,
                        "analysis": analysis
                    }
                st.success(f"{uploaded_file.name} analyzed successfully!")
        
    # Reset Chat History
    if st.button("Reset Chat History"):
        st.session_state.chat_history = []

# Main chat interface
st.title("FinBot")
st.markdown("Welcome to FinBot! Please enter your financial question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about the financial reports?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.pdf_analyses:
            llm = get_llm_hf_inference()
            chain = get_conversational_chain(llm)
            
            # Prepare context from all analyzed PDFs
            context = ""
            for file_id, file_data in st.session_state.pdf_analyses.items():
                context += f"Report: {file_data['name']}\n"
                context += f"Analysis: {file_data['analysis']['analysis']}\n"
                context += f"Review: {file_data['analysis']['review']}\n"
                context += f"Recommendation: {file_data['analysis']['recommendation']}\n\n"

            response = chain.run(context=context, human_input=prompt)
        else:
            response = "I'm sorry, but no financial reports have been uploaded and analyzed yet. Please upload PDF financial reports first."
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display uploaded and analyzed PDFs
if st.session_state.pdf_analyses:
    st.sidebar.header("Analyzed Reports")
    for file_id, file_data in st.session_state.pdf_analyses.items():
        st.sidebar.text(file_data['name'])
