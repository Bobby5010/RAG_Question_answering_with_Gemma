
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
import sentence_transformers
from data_utils import load_queries


queries = load_queries()

#func to combine a list of langchain Documents (combine the metadata and the page source)
def combine_docs(docs):
    page_content = ''
    section = set(); page = set()
    for doc in docs:
        page_content = page_content + "\n\n" + doc[0].page_content
        section.add(doc[0].metadata['sections'])
        page.add(doc[0].metadata['pages'])
    document = Document(
        metadata = {'sections' : list(section), 'pages' : list(page)},
        page_content = page_content  
    )
    return document


def prepare_retrieval_sources(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    
    #split the documents into small chunks for better retrival od relevant documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 100 
    ) 
    
    split_docs = text_splitter.split_documents(docs)
    
    faiss_vs = FAISS.from_documents(split_docs, embeddings)
    faiss_sources = [faiss_vs.similarity_search_with_score(query, k = 4) for query in queries]
    
    #combine the Nearest neighbour documents obtained from above sources to a single langchain Document Object
    sources = []
    for source in faiss_sources:
        sources.append(combine_docs(source))
    return sources
