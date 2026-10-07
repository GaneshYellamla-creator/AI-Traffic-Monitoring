from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

VECTOR_DIR = "rag/vector_db"


class TrafficRAG:

    def __init__(self):

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.db = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings
        )

    def search(self, query, k=3):

        documents = self.db.similarity_search(
            query,
            k=k
        )

        return documents

    def get_context(self, query):

        documents = self.search(query)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        return context