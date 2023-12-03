from typing import List, Optional
import os
import pdfplumber
from bs4 import BeautifulSoup
from pytesseract import image_to_string
from PIL import Image
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
import logging
import json
import logging.config
import logging.handlers
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import tiktoken
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Configure logging to file
logging.config.fileConfig('parameters/logs.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class DataLoader(BaseLoader):
    """Load various text documents."""

    def __init__(self, file_path: str, encoding: Optional[str] = None, autodetect_encoding: bool = False):
        """Initialize with file path."""
        self.logger = logger
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
        self.logger.info(f"DataLoader initialized for file: {file_path}")

    def load(self) -> List[Document]:
        """Load from file path based on file type."""
        self.logger.info(f"Starting File Load Process: {self.file_path}")
        file_extension = os.path.splitext(self.file_path)[1].lower()
        try:
            if file_extension in ['.txt']:
                return self._load_text_file()
            elif file_extension in ['.pdf']:
                return self._load_pdf_file()
            elif file_extension in ['.html', '.htm']:
                return self._load_html_file()
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff']:
                return self._load_image_file()
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        except Exception as e:
            self.logger.error(f"Error loading file {self.file_path}: {e}")
            raise

    def _load_text_file(self):
        """Load a text file."""
        self.logger.info(f"Loading text file: {self.file_path}")
        text = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except Exception as e:
            self.logger.error(f"Error loading text file {self.file_path}: {e}")
            raise RuntimeError(f"Error loading {self.file_path}: {e}")

        metadata = {"source": self.file_path, "type": "text"}
        return [Document(page_content=text, metadata=metadata)]

    def _load_pdf_file(self):
        """Load a PDF file."""
        self.logger.info(f"Loading PDF file: {self.file_path}")
        text = ""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            self.logger.error(f"Error loading PDF file {self.file_path}: {e}")
            raise RuntimeError(f"Error loading {self.file_path}: {e}")

        metadata = {"source": self.file_path, "type": "pdf"}
        return [Document(page_content=text, metadata=metadata)]

    def _load_html_file(self):
        """Load an HTML file."""
        self.logger.info(f"Loading HTML file: {self.file_path}")
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text()
        except Exception as e:
            self.logger.error(f"Error loading HTML file {self.file_path}: {e}")
            raise RuntimeError(f"Error loading {self.file_path}: {e}")

        metadata = {"source": self.file_path, "type": "html"}
        return [Document(page_content=text, metadata=metadata)]

    def _load_image_file(self):
        """Load text from an image file using OCR."""
        self.logger.info(f"Loading image file: {self.file_path}")
        try:
            text = image_to_string(Image.open(self.file_path))
        except Exception as e:
            self.logger.error(f"Error loading image file {self.file_path}: {e}")
            raise RuntimeError(f"Error loading {self.file_path}: {e}")

        metadata = {"source": self.file_path, "type": "image"}
        return [Document(page_content=text, metadata=metadata)]


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.params = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path) as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

    def get_api_key(self):
        return self.params.get('GPT', {}).get('API', None)

class Logger:
    def __init__(self, config_file):
        logging.config.fileConfig(config_file, disable_existing_loggers=False)
        self.logger = logging.getLogger(__name__)

    def get_logger(self):
        return self.logger




class EmbeddingManager:
    def __init__(self, api_key, logger):
        self.api_key = api_key
        self.logger = logger

    def create_embeddings(self, chunks):
        """
        Create embeddings for the given chunks of text.
        :param chunks: List of text chunks to be embedded.
        :return: A vector store containing the embeddings.
        """
        try:
            self.logger.info("Creating embeddings...")
            embeddings = OpenAIEmbeddings(api_key=self.api_key)
            vector_store = Chroma.from_documents(chunks, embeddings)
            self.logger.info("Embeddings created successfully.")
            return vector_store
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {e}")
            raise

    def calculate_embedding_cost(self, texts):
        """
        Calculate the cost of creating embeddings for the given texts.
        :param texts: List of text documents for which to calculate the embedding cost.
        :return: Tuple of total tokens and the calculated cost.
        """
        try:
            self.logger.info("Calculating embedding cost...")
            encoder = tiktoken.encoding_for_model('text-embedding-ada-002')

            # Calculate total tokens
            total_tokens = sum([len(encoder.encode(text)) for text in texts])
            cost = total_tokens / 1000 * 0.0004  # Example cost calculation

            self.logger.info(f'Total Tokens: {total_tokens}')
            self.logger.info(f'Embedding Cost in USD: {cost:.6f}')
            return total_tokens, cost
        except Exception as e:
            self.logger.error(f"Error in calculating embedding cost: {e}")
            raise


class QueryProcessor:
    def __init__(self, api_key, logger):
        self.api_key = api_key
        self.logger = logger

    def process_query(self, vector_store, query, k):
        """
        Process the user query and retrieve an answer.

        :param vector_store: The vector store containing document embeddings.
        :param query: The user's query as a string.
        :param k: Number of top documents to retrieve for answering the query.
        :return: The answer to the query.
        """
        try:
            self.logger.info(f"Processing query: {query}")
            # Set up the language model and retriever
            llm = ChatOpenAI(api_key=self.api_key, model_name='gpt-4-1106-preview', temperature=1)
            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

            # Create and run the retrieval chain
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
            answer = chain.run(query)
            self.logger.info("Query processed successfully.")
            return answer
        except Exception as e:
            self.logger.error(f"Error in processing query: {e}")
            raise