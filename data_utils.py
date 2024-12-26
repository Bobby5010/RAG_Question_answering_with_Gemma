
from langchain_community.document_loaders import PyPDFLoader
from copy import deepcopy
import pandas as pd

file_path = 'data/book.pdf'
loader = PyPDFLoader(file_path)
pages = loader.load()


#loads the corpus of text to search in for RAG
def get_contents():
    return pages[6:11]

def get_contexts():
    return pages[18:]
    
#load the queries to be answered
def load_queries():
    query_df = pd.read_json('data/queries.json')
    queries = [query for query in query_df['question']]
    return queries


#unwanted sections of the book to be removed later 
def unwanted_contexts():
    return {'Key Terms','Summary','Review Questions','Critical Thinking Questions','Personal Application Questions','References','Index'}

#split the contents based on chapters
def split_contents():   
    content_docs = get_contents()
    contents = sum([content.page_content.split("\n") for content in content_docs], [])
    chapter_contents = [[]]
    for content in contents:
        if content.startswith("CHAPTER"):
            chapter_contents.append([])
            continue
        chapter_contents[-1].append(content)
    return chapter_contents


#func to make the sections as "chapter_name/subtopic_name"
def make_sections(chapter_content):
    chapter_name = " ".join(chapter_content[0].split()[:-1])
    contents = [f"{chapter_name}/{chapter_content[1]}"]
    for section in chapter_content[2:]:
        try:
            split = section.split()
            float(split[0])
            contents.append(f"{chapter_name}/{' '.join(split[1:])}")
        except ValueError:
            try:
                int(split[-1])
                contents.append(' '.join(section.split()[:]))
            except ValueError:
                pass
    return contents


def extract_corpus():
    chapter_contents = split_contents()
    sections = []
    for chapter_content in chapter_contents[1:]:
        sections.extend(make_sections(chapter_content)) 
    
    #map the page_numbers with the corresponding topic name
    ps_map = {}
    for section in sections:
        split = section.split()
        page = int(split[-1])
        title = ' '.join(split[:-1])
        if page in ps_map:
            ps_map[page].append(title)
        else :
            ps_map[page] = [title]
    

    context_docs = get_contexts()
    unwanted_sections = unwanted_contexts()
    #Remove the unwanted sections of the context like Key-Terms, Practise Sections etc and store  the relevannt docs in a list  
    docs = []
    for context_doc in context_docs:
        doc = deepcopy(context_doc)
        page = doc.metadata['page']-11
        while page not in ps_map: 
            page-=1
        if ps_map[page][0] not in unwanted_sections: 
            doc.metadata = {"sections" : ps_map[page][0] , 'pages' : page, 'page' : doc.metadata['page']-11}
            docs.append(doc)
            
    return docs
