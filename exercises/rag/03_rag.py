from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.storage import LocalFileStore
from utils import get_attendance_id
from pathlib import Path

VECTOR_STORE_PATH = '.cache/vector_store.p'

# Exercise 3 - Full RAG Pipeline
# This exercise implements a full RAG (Retrieval-Augmented Generation) pipeline.
#   - It depends on the vector store created in exercise 2! Make sure you have completed that first.
#
# Part 1
#   * Read through the code and run it.
#   * It will print an answer to the question based on the documents in the `data/` folder.
#   * Uncomment the print statement to see the full prompt sent to the LLM.
#
# Part 2
#   * Think about the prompt template. Are there any issues with it?
#   * What happens if the retrieved documents are irrelevant?
#

# Embeddings model
embeddings = BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v2:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)

print('Computing embeddings for documents...')

# Use cached embeddings where possible
store = LocalFileStore("./.cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store,
    key_encoder='sha256'
)

if not Path.exists(Path(VECTOR_STORE_PATH)):
    raise FileNotFoundError("Vector store not found! Please run exercise 2 first to create it.")

# Load vector store

if not Path.exists(Path(VECTOR_STORE_PATH)):
    print('Vector store not found! Please run exercise 2 first to create it.')
    exit(1)

print('Loading existing vector store...')
vector_store = InMemoryVectorStore.load(VECTOR_STORE_PATH, cached_embeddings)

# Create LLM
llm = ChatBedrock(
    model='eu.amazon.nova-pro-v1:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)

prompt = ChatPromptTemplate(
    [
        ("human", """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
                Question: {question}
                Context: {context}
                Answer:
         """)
    ]
)

QUESTION = "Has Clamazon been involved in any controversies?"

# Retrieve relevant documents from the vector store and construct the prompt
retrieved_docs = vector_store.similarity_search(QUESTION)

DOCS_CONTENT = "\n\n".join(doc.page_content for doc in retrieved_docs)
qa_prompt = prompt.invoke({"question": QUESTION, "context": DOCS_CONTENT})

# Uncomment to see the full prompt sent to the LLM
# print('*' * 50)
# print('Full prompt sent to LLM:', end='')
# print(qa_prompt.messages[0].content)
# print('*' * 50)

answer = llm.invoke(qa_prompt)
print()
print('=' * 50)
print(f'Question: {QUESTION}')
print('-' * 50)
print(f'LLM Answer: {answer.content}')
