# Advanced Multi-Document Communication Enabled Using LLM and Google Gemini

This is a Streamlit-based application that allows users to upload multiple PDF and DOCX files, process them for text extraction, and ask questions to retrieve answers from the documents. It leverages Google Generative AI for text embeddings, conversational AI, and document summarization. The app also supports translation of answers into multiple languages.

## Features
- Upload multiple PDF and DOCX files.
- Extract text from uploaded documents.
- Generate document summaries.
- Answer questions based on the content of the documents.
- Translate answers into multiple languages (e.g., French, Spanish, German, Chinese, Telugu, Hindi, Tamil).

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Web framework for building the user interface.
- **PyPDF2**: For extracting text from PDF files.
- **python-docx**: For extracting text from DOCX files.
- **LangChain**: Framework for conversational AI and question-answering.
- **Google Generative AI**: Used for embeddings, conversational AI, and summarization.
- **FAISS**: For vector similarity search.
- **Deep Translator**: For language translation.

## Prerequisites
1. Python 3.8 or above installed.
2. Install required Python libraries (see the **Installation** section below).
3. A Google Generative AI API key (set up in your `.env` file).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-document-answering.git
   cd multi-document-answering
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file:
   ```plaintext
   GOOGLE_API_KEY=your_google_api_key
   ```

## How to Run
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the app in your browser at `http://localhost:8501`.

## Usage
1. Upload PDF or DOCX files through the sidebar menu.
2. Click the **Submit & Process** button to extract text and generate summaries.
3. View the summaries of each uploaded document.
4. Ask questions in the text input box, and view detailed responses.
5. Optionally, translate the responses into a selected language.

## File Structure
- `app.py`: Main application script.
- `requirements.txt`: List of Python dependencies.
- `.env`: File for storing API keys (not included in the repository for security).

## Dependencies
- streamlit
- PyPDF2
- langchain
- google-generativeai
- python-docx
- FAISS
- deep-translator
- python-dotenv

## Screenshots
### Upload Documents
*(Include a screenshot showing the document upload feature)*

### Document Summaries
*(Include a screenshot showing document summaries)*

### Q&A and Translations
*(Include a screenshot showing the Q&A feature with translations)*

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [Google Generative AI](https://ai.google/) for providing advanced AI capabilities.
- [Streamlit](https://streamlit.io/) for enabling quick web application development.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
