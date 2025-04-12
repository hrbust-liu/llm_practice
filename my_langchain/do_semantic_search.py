from langchain_core.documents import Document

documents = [
    Document(
        page_content="我家有个狗, 叫旺财.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="我家有个猫，叫招财.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
# from langchain_community.document_loaders import PyPDFLoader

# file_path = "/root/langchain/docs/docs/example_data/nke-10k-2023.pdf"
# loader = PyPDFLoader(file_path)

# docs = loader.load()

# print(len(docs))

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(documents) # docs

print("allsplits: %s" % len(all_splits))

# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# # vector_2 = embeddings.embed_query(all_splits[1].page_content)

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2"
#   model_name="qwen-plus",
# model_kwargs={"token": "your_huggingface_token"}  # 需替换为实际 Token
)
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "我家的猫，叫什么"
)

print(results[0])
print(f"results: {results}, len: {len(results)}")

