
# Language Model-Based Question-Answering Application
![app.png](assets%2Fapp.png)
## Overview
This project is a Streamlit-based web application designed to provide question-answering capabilities using language models. 
It allows users to upload documents, process them, and use a language model to answer questions based on the content of these documents. 
At the moment application leverages OpenAI's GPT model for generating answers.
Llama2 70b and other LLMs are going to be implemented as well. 

![diagram.png](assets%2Fdiagram.png)

## Features
- **Document Upload**: Users can upload documents in various formats (PDF, DOCX, TXT).
- **Text Processing**: The application chunks the text data for efficient processing.
- **Embeddings Creation**: It generates embeddings for the uploaded text using OpenAI's APIs.
- **Question Answering**: Users can query the uploaded document, and the app uses a language model to provide relevant answers.

## Project Structure
```angular2html
LLM-LangChain-Streamlit
├── assets
│   ├── diagram.png
│   └── q.png
├── data
│   └── TroutStocking.pdf
├── lib
│   ├── core.py
│   ├── __init__.py
│   └── __pycache__
├── logs
│   └── Process.log
├── models
├── notebooks
│   ├── Custom ChatGPT with LangChain.ipynb
│   ├── FullProject.ipynb
│   ├── GPT-4.ipynb
│   ├── Llama2.ipynb
│   ├── OpenAIAssistant.ipynb
│   ├── PineCone.ipynb
│   └── tools.ipynb
├── OpenAI_Chroma_LangChain_Strimlit.py
├── parameters
│   ├── config_template.json
│   └── logs.ini
├── README.md
├── setup.py
└── requirements.txt
```

## Requirements
- streamlit
- langchain
- pdfplumber
- BeautifulSoup
- pytesseract
- PIL
- tiktoken
- dotenv

## Setup and Installation
1. Clone the repository:
   ```
   git clone https://github.com/SlavPetkovic/LLM-LangChain-Streamlit.git
   ```
2. Navigate to the project directory:
   ```
   cd LLM-LangChain-Streamlit
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Set up your `.env` file with the necessary environment variables, such as your OpenAI API key.

## Usage
1. Start the Streamlit app:
   ```
   streamlit run OpenAI_Chroma_LangChain_Strimlit.py
   ```
2. Open your web browser and go to `localhost:8501` (or the URL provided in the terminal).
3. Input your OpenAI API key in the sidebar.
4. Upload a document and set the desired chunk size and `k` value for the query.
5. After processing, input your question in the provided text box to get answers based on the document's content.

## Logging
Logs are written to a file as configured in `parameters/logs.ini`. You can review these logs for debugging and monitoring the application's performance.

## Contributing
Contributions to this project are welcome. Please follow the standard fork-pull request workflow.

# Upcoming:
- Adding Llama 2 70b lanaguage as an optional LLM within app
- Adding pinecone as optional VectorDB within App
- Adding code for OpenAI and Llama2 debate sessions

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
