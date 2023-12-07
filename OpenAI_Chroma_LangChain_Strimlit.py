# Import dependencies
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from lib.core import DataLoader, Config, Logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging.config
import logging.handlers


logger = Logger('parameters/logs.ini').get_logger()

def chunk_data(data, chunk_size= 256, chunk_overlap=20):
    try:
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap= chunk_overlap,
            length_function=len)
        chunks = text_splitter.create_documents(text_contents)
        logging.info(f"Data chunking successful. Number of chunks: {len(chunks)}")
        return chunks
    except Exception as e:
        logging.error(f"Error in chunking data: {e}")

def create_embeddings(chunks):
    try:
        logger.info("Creating embeddings...")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = Chroma.from_documents(chunks, embeddings)
        logger.info("Embeddings created successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
def prompt(vector_store, query,k):
    try:
        logger.info(f"Processing query: {query}")
        # Set up the language model and retriever
        llm = ChatOpenAI(api_key=api_key, model_name='gpt-4-1106-preview', temperature=1)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
        # Create and run the retrieval chain
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        answer = chain.run(query)
        logger.info("Query processed successfully.")
        return answer
    except Exception as e:
        logger.error(f"Error in processing query: {e}")

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def embedding_cost(texts):
    try:
        logger.info("Calculating embedding cost...")
        # Import tiktoken module
        import tiktoken
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        # Calculate total tokens
        total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
        cost = total_tokens / 1000 * 0.0004
        # Log the calculated cost
        logger.info(f'Total Tokens: {total_tokens}')
        logger.info(f'Embedding Cost in USD: {cost:.6f}')
        return total_tokens, cost
    except Exception as e:
        logger.error(f"Error in calculating embedding cost: {e}")

if __name__ == "__main__":
    import os
    import tempfile
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    logging.basicConfig(level=logging.ERROR)

    st.image('assets/q.png')
    st.subheader('LLM Question-Answering Application')

    # Initialize history in session state if not present
    if 'history' not in st.session_state:
        st.session_state.history = ''

    with (st.sidebar):
        api_key = st.text_input('Your OpenAI API Key', type='password')
        if api_key:
            os.environ['OPEN_AI_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk Size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file...'):
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name

                # Use the temporary file path with DataLoader
                loader = DataLoader(file_path=temp_file_path)
                data = loader.load()

                try:
                    text_contents = [doc.page_content for doc in data]
                except Exception as e:
                    logging.error(f"Error processing document contents: {e}")

                chunks = chunk_data(text_contents, chunk_size=chunk_size)
                st.write(f'Chunk Size:{chunk_size}, Chunks:{len(chunks)}')

                tokens, embedding_cost = embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("File Uploaded, Chunked and Embedded Successfully.")

    query = st.text_input("Ask the question about the content of your file")

    if query:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = prompt(vector_store, query, k)
            st.text_area('LLM Answer:', value=answer, height=300)

            st.divider()
            # Update history in session state
            value = f'Q:{query} \nA: {answer}'
            if 'history' in st.session_state:
                st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            else:
                st.session_state.history = value

            # Use session state for widget value, without a default
            st.text_area(label='Chat History', value=st.session_state.history if 'history' in st.session_state else '',
                         key='history', height=400)
