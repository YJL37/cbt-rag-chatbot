from langchain_google_genai import ( GoogleGenerativeAIEmbeddings )
from langchain_chroma import Chroma
import chromadb
from uuid import uuid4


class VectorDBManager:
    def __init__(self, embedding_model="models/text-embedding-004"):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.client = chromadb.PersistentClient()

    def init_db(self, collection_name):
        # create collection
        print("==== Initializing database with collection: " + collection_name + " ====")
        existing_collections = self.client.list_collections()
        if any(
            collection.name == collection_name
            for collection in existing_collections
        ):
            print(
                f"Collection '{collection_name}' already exists. Skipping initialization process."
            )
        else:
            vector_store = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding_model,
            )

    def upload_docs(self, docs, collection_name):
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents = docs, ids=uuids)
        print(
            f"Added documents to collection " + collection_name
        )



# ChromaDB

# "client": chroma DB 전체 관리
# "collection": table - dataset 종류 (cbt, 심리상담, qna)

# cbt collection