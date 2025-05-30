import os
import streamlit as st
import validators
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Supported languages
top_languages = [
    ("English", "en"), ("Mandarin Chinese", "zh"), ("Hindi", "hi"),
    ("Spanish", "es"), ("French", "fr"), ("Standard Arabic", "ar"),
    ("Bengali", "bn"), ("Portuguese", "pt"), ("Russian", "ru"),
    ("Japanese", "ja"), ("Gujarati", "gu")
]

titles = {
    "English": "🧠 Smart Web Summarizer",
    "Hindi": "🧠 स्मार्ट वेब सारांशकर्ता",
    "Spanish": "🧠 Resumen Inteligente de Sitios Web",
    "French": "🧠 Résumeur Web Intelligent",
    "Standard Arabic": "🧠 الملخّص الذكي للمواقع الإلكترونية",
    "Mandarin Chinese": "🧠 智能网页摘要工具",
    "Bengali": "🧠 স্মার্ট ওয়েব সারাংশকারী",
    "Portuguese": "🧠 Resumidor Inteligente de Sites Web",
    "Russian": "🧠 Умный Суммаризатор Веб-сайтов",
    "Japanese": "🧠 スマートウェブ要約ツール",
    "Gujarati": "🧠 સ્માર્ટ વેબ સારાંશ સાધન"
}

# Set page config
st.set_page_config(page_title="🌐 Website Summarizer", page_icon="🧠")

# --- User Inputs ---
language_name = st.selectbox("🌍 Select summary language:", [name for name, _ in top_languages])
language_code = dict(top_languages)[language_name]
st.title(titles.get(language_name, "🧠 Smart Web Summarizer"))

url = st.text_input("🔗 Enter a Website URL:")
model_name = st.selectbox("🤖 Choose a Groq model:", [
    "gemma2-9b-it", "Llama-3.1-8b-Instant", "Llama3-8b-8192",
    "Llama-3.3-70b-Versatile", "DeepSeek-R1-Distill-Llama-70B"
])

# --- Function to initialize LLM ---
def get_llm(model_name):
    return ChatGroq(model=model_name)

# --- Generate Summary ---
if st.button("🚀 Generate Summary"):
    if not url.strip():
        st.error("Please enter a website URL.")
    elif not validators.url(url):
        st.error("The input is not a valid URL.")
    else:
        try:
            with st.spinner("Fetching and processing content..."):
                llm = get_llm(model_name)

                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/116.0.0.0 Safari/537.36"
                    }
                )
                raw_docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(raw_docs)

                map_prompt = PromptTemplate(
                    template="""
You are a helpful assistant. Extract the key points from the following content. Respond in {language}. Avoid internal thoughts.

TEXT:
{text}
""",
                    input_variables=["text", "language"]
                )

                combine_prompt = PromptTemplate(
                    template="""
You are a helpful assistant. Using the key points below, write a clear and concise summary in {language}. Avoid internal thoughts.

KEY POINTS:
{text}

Summary:
""",
                    input_variables=["text", "language"]
                )

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                    verbose=False
                )

                summary = chain.run({"input_documents": docs, "language": language_name})
                st.subheader(f"📄 Summary in {language_name}:")
                st.success(summary)

        except Exception as e:
            st.error("An error occurred during summarization.")
            st.exception(e)
