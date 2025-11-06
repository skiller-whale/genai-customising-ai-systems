from pathlib import Path
from time import perf_counter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from utils import get_attendance_id

VECTOR_STORE_PATH = '.cache/vector_store.p'

# Exercise 2 - Vector Store
#
# This exercise creates a vector store from a set of documents.
#
#  * Before you begin, skim through the documents in the `data/` folder to get a sense of what they contain.
#
# Part 1 - Creating the Vector Store
#   * Read through the code and run it.
#   * It will create a vector store and save it to `.cache/vector_store.p`.
#
# Part 2 - Similarity Search
#   * Uncomment the lines at the end of the file to run a similarity search.
#       - It will print the top 3 most similar document chunks to the query.
#   * Change the query and see what results you get.

# load documents from data/
loader = DirectoryLoader("./data/", glob="*.txt")
docs = loader.load()

# Split documents in chunks
#   paragraphs are roughly 500 characters each with a chunk overlap 100
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)

# Embeddings model
embeddings = BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v2:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)

print('Computing embeddings for documents...')

# Cache embeddings to avoid recomputing if a chunk was already processed
store = LocalFileStore("./.cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, key_encoder='sha256'
)

# Load Vector Store if it exists, otherwise create it
if Path.exists(Path(VECTOR_STORE_PATH)):
    print('Loading existing vector store...')
    vector_store = InMemoryVectorStore.load(VECTOR_STORE_PATH, cached_embeddings)
else:
    print('Creating new vector store...')
    start_time = perf_counter()
    vector_store = InMemoryVectorStore(cached_embeddings)
    _vidx = vector_store.add_documents(documents=all_splits)
    print(f'Creating VS took: {perf_counter() - start_time:.3f}s')

    vector_store.dump(VECTOR_STORE_PATH)

# Uncomment the following lines to run a similarity search, returning the 3 best results.
#
# QUERY = "What is Clamazon's business model?"
# results = vector_store.similarity_search(QUERY, k=3)
# for i, r in enumerate(results, start=1):
#     print(f'--- Result {i} ---')
#     print(r.page_content)
#     print()
