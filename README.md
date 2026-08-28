# RAG-LAW-FINAL 🏛️

A powerful **Retrieval-Augmented Generation (RAG)** system for querying and understanding legal documents using advanced AI models. This project combines LLMs with legal document processing to provide intelligent, context-aware responses to legal queries.

## 🎯 Overview

This project implements a RAG pipeline that processes legal documents (PDFs) and enables users to ask questions about them using state-of-the-art language models. It currently supports documents including:

- **Bharatiya Nagarik Suraksha Sanhita, 2023** (Indian Criminal Code)
- **Universal Declaration of Human Rights**

## ✨ Features

- 📄 **PDF Document Processing**: Efficiently parse and extract content from legal PDFs
- 🔍 **Semantic Search**: Find relevant sections using vector embeddings
- 🤖 **AI-Powered Responses**: Get accurate answers using DeepSeek R1 or other LLMs
- 💾 **Vector Storage**: Persistent vector database for fast retrieval
- 🌐 **Web Interface**: User-friendly Streamlit interface for easy interaction
- 🔗 **Context Preservation**: Maintains document context for accurate legal interpretations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aayush-2701/RAG-LAW-FINAL.git
   cd RAG-LAW-FINAL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Run the Web Interface
```bash
streamlit run app.py
```

The application will open in your browser, allowing you to:
- Upload legal documents (PDFs)
- Ask questions about the documents
- View retrieved relevant sections
- Get AI-generated answers with citations

#### Use the RAG Pipeline Directly
```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline()

# Query a document
response = rag.query("What are the punishment provisions?")
print(response)
```

## 📁 Project Structure

```
RAG-LAW-FINAL/
├── app.py                           # Streamlit web application
├── rag_pipeline.py                  # Core RAG implementation
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python runtime specification
├── Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf    # Indian Criminal Code
├── universal_declaration_of_human_rights.pdf       # Human Rights document
├── Vectorstore/                     # Vector database storage
└── streamlit/                       # Additional Streamlit configs
```

## 🛠️ Technology Stack

- **Language Models**: DeepSeek R1, Open-source LLMs
- **Vector Database**: Faiss/Chroma for embeddings storage
- **Document Processing**: PyPDF2, LangChain
- **Web Framework**: Streamlit
- **Embeddings**: Hugging Face Transformers
- **Python Libraries**: See `requirements.txt`

## 📋 Features in Detail

### Document Processing
- Extracts text from PDF documents
- Splits documents into manageable chunks
- Generates vector embeddings for semantic search

### Retrieval Mechanism
- Uses semantic similarity for retrieving relevant sections
- Returns top-k most relevant chunks based on query
- Preserves document structure and context

### Generation
- Passes retrieved context to LLM
- Generates accurate, context-aware responses
- Includes source citations for transparency

## 🔧 Configuration

### Dependencies

Key packages (see `requirements.txt` for full list):
- `streamlit` - Web interface
- `langchain` - RAG framework
- `faiss-cpu` or `chroma` - Vector stores
- `sentence-transformers` - Embeddings
- `transformers` - LLM support
- `pypdf` - PDF processing

### Environment Variables

Create a `.env` file if needed for API keys:
```bash
# Add your API keys here
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

## 📊 How It Works

```
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Retrieve Relevant Documents
    ↓
Create Prompt with Context
    ↓
LLM Generation
    ↓
Return Answer with Citations
```

## 🎓 Use Cases

- **Legal Research**: Quickly find relevant clauses and sections
- **Document Analysis**: Understand complex legal language
- **Case Preparation**: Research applicable laws and rights
- **Educational**: Learn about legal documents and their provisions
- **Compliance**: Check regulatory requirements

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Add support for more legal documents

## 📝 License

This project is open source. Please check the repository for license details.

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DeepSeek Documentation](https://www.deepseek.com/)
- [Bharatiya Nagarik Suraksha Sanhita, 2023](https://www.indiacode.nic.in/)

## 🔐 Legal Notice

This tool is designed for informational purposes. While it provides AI-assisted analysis of legal documents, it should not replace professional legal advice. Always consult with qualified legal professionals for matters requiring legal interpretation.

## 💬 Support & Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the repository maintainer.

---

**Built with ❤️ for legal research and AI-powered document analysis**

Last Updated: 2026-08-28
