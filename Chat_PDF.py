import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import requests
from streamlit_lottie import st_lottie
import json
import time  # Import the time module

# Sidebar contents
with st.sidebar:
    st.title('Chat with PDF')
    st.markdown('''
        # Name:
        Pratiksha Badgujar
        - [Portfolio](https://pratikshabadgujar.vercel.app/)
        - Contact: pratikshasbadgujar@gmail.com
        - Phone number: 7499688988
    ''')

    add_vertical_space(10)
    st.write('Made with ❤ by Pratiksha Badgujar')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
    st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        height=None,
        width=None,
        key=None)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            try:
                docs = VectorStore.similarity_search(query=query, k=3)

                # Update the model name to gpt-3.5-turbo
                llm = OpenAI(model_name="gpt-3.5-turbo")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.write(response)
            except RateLimitError as e:
                st.error(f"Rate limit exceeded. Please wait and try again later.")
                # Optionally, you can implement a retry mechanism with backoff here.
                time.sleep(60)  # Wait for 60 seconds before retrying.

if __name__ == '__main__':
    main()
