import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

def test_search():
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

  from langchain_community.embeddings import HuggingFaceEmbeddings
  embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
  from langchain_core.vectorstores import InMemoryVectorStore

  vector_store = InMemoryVectorStore(embeddings)
  ids = vector_store.add_documents(documents=documents)

  results = vector_store.similarity_search(
      "我家的猫，叫什么"
  )

  print(results[0])
  print(f"results: {results}, len: {len(results)}")


def test_search_v2():
  from langchain_core.documents import Document

  documents = [
      Document(
          page_content="XMK 成绩 100",
          metadata={"source": "mammal-pets-doc"},
      ),
      Document(
          page_content="FXK 成绩 120",
          metadata={"source": "mammal-pets-doc"},
      ),
      Document(
          page_content="YS 成绩 90",
          metadata={"source": "mammal-pets-doc"},
      ),
  ]
  from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2"
  )
  from langchain_core.vectorstores import InMemoryVectorStore

  vector_store = InMemoryVectorStore(embeddings)

  # ids = vector_store.add_documents(documents=all_splits)
  ids = vector_store.add_documents(documents=documents)

  results = vector_store.similarity_search(
      "谁的成绩最好"
  )

  print(results[0])
  print(f"results: {results}, len: {len(results)}")

test_search_v2()