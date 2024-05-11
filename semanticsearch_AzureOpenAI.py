import os
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

load_dotenv()

# Configure Azure OpenAI Service API
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
API_VERSION = "2024-02-01"

file_path = 'Files/test.txt'  # replace with your text file path

# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CHUNK_SIZE = 8000
COMPLETION_MODEL = 'text-davinci-003'
client = AzureOpenAI(api_key=API_KEY,azure_endpoint=API_ENDPOINT,api_version=API_VERSION)  
ef = embedding_functions.DefaultEmbeddingFunction()
#db_path = 'Files/chroma_db'  # replace with your Chroma DB path
chroma_client = chromadb.PersistentClient(path='Files/chroma_db')

def load_data_from_file():
    """
    Load data from a text file into a list of strings
    """
    with open(file_path, 'r',encoding="utf8") as f:
        data = [line.strip() for line in f.readlines()]
    return data

def create_chroma_db( data) -> chromadb.Collection:
    """
    Create a Chroma vector database and add data to it
    """ 
    chroma_collection = chroma_client.get_or_create_collection(name="chroma_collection",embedding_function=ef)   
       
    docid =[]
    i=0
    for i,dataa in enumerate(data):               
        docid.append(f"doc_{i}")
        
    chroma_collection.upsert(documents=data, ids= docid)
    return chroma_collection

def search_semantically(chrcoll:chromadb.Collection, query):
    """
    Perform semantic search using the Chroma vector database
    """    
    
    results = chrcoll.query(query_texts=query)
    
    return results

def main(): 
    
    data = load_data_from_file()
    ChrCollection = create_chroma_db( data)
    querylist =[]
    query = "what is azure?" #input("Enter your search query: ")
    querylist.append(query)
    results = search_semantically(ChrCollection,querylist)
    print("Search results:")   
    print(results["documents"])
   
if __name__ == '__main__':
    main()