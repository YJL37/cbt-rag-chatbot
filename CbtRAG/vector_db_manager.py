from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from uuid import uuid4
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.retrievers.multi_query import MultiQueryRetriever
import ast
from langchain import hub
from langchain.tools.retriever import create_retriever_tool

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


class VectorDBManager:
    def __init__(self, embedding_model="models/text-embedding-004"):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.client = chromadb.PersistentClient()

    def init_db(self, collection_name):
        # create collection
        print("    Initializing database with collection: " + collection_name + "...")
        existing_collections = self.client.list_collections()
        if any(
            collection.name == collection_name for collection in existing_collections
        ):
            print(
                f"    Collection '{collection_name}' already exists. Deleting existing collection."
            )
            self.client.delete_collection(collection_name)

        # create new collection (table)
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )

        print(f"    Successfully created new collection '{collection_name}'")

        return

    def upload_docs(self, docs, collection_name):
        # error handling: collection name doesn't exist in client
        print("    Uploading documents to collection: " + collection_name + "...")
        existing_collections = self.client.list_collections()
        if not any(
            collection.name == collection_name for collection in existing_collections
        ):
            raise ValueError("    Collection name doesn't exist in client")

        # get collection
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
        print(f"    Successfully uploaded documents to '{collection_name}'")

        return

    # Context Retrieval --------------------------------------------------------
    def retrieve_contexts(self, query, collection_name):
        # 어떻게 retrieve...
        existing_collections = self.client.list_collections()
        if not any(
            collection.name == collection_name for collection in existing_collections
        ):
            print("error handling: collection name doesn't exist in client")
            return False

        # get collection
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )

        # override the default similarity search function to display scores
        @chain
        def retriever(query: str) -> List[Document]:
            docs, scores = zip(*vector_store.similarity_search_with_score(query, k=4))
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score

            return docs

        contexts = retriever.invoke(query)

        return contexts

    # Query Expansion (Multiple Queries)
    def retrieve_contexts_with_multi_queries(self, query, collection_name, llm):
        # query -> llm -> multiple queries -> retrieval contexts
        # user가 고민을 털어놨을 때 => 심리적인 이유, 가족, 친구...

        # 1. query => llm => multiple queries
        output_parser = LineListOutputParser()
        # TODO fine-tune this prompt
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI psychological counselor. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        # query generation chain
        llm_chain = QUERY_PROMPT | llm | output_parser

        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )

        retriever = MultiQueryRetriever(
            retriever=vector_store.as_retriever(),
            llm_chain=llm_chain,
            parser_key="lines",
        )

        contexts = retriever.invoke(query)

        # deduplicate contexts
        unique_documents = set()
        for context in contexts:
            unique_documents.add(document_to_str(context))

        final_contexts = [str_to_document(doc) for doc in unique_documents]
        print(f"Retrieved Contexts: {len(final_contexts)}")

        return final_contexts

        # 2. multiple queries => retrieve context from collection

    # Tool Creation ------------------------------------------------------------
    def create_tool_with_collection(self, database):
        # get collection
        vector_store = Chroma(
            client=self.client,
            collection_name=database["dataset_name"],
            embedding_function=self.embedding_model,
        )

        retriever = vector_store.as_retriever()

        tool = create_retriever_tool(
            retriever=retriever,
            name=database["tool_name"],
            description=database["tool_description"],
        )

        database_format = database["format"]

        def format_contexts(docs):
            # docs: str
            formatted_contexts = database_format.format(context=docs)
            return formatted_contexts

        # Here, we can add context reranking, compression, etc by chaining

        retriever_tool = tool | format_contexts

        # return retriever_tool.as_tool()
        return tool

    def create_rag_chain_tool_with_collection(self, collection_name, llm):
        # get collection
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )

        system_prompt = """
            You are an assistant for question-answering tasks.
            Use the below context to answer the question. If
            you don't know the answer, say you don't know.
            Use three sentences maximum and keep the answer
            concise.

            Question: {question}

            Context: {context}
            """

        prompt = hub.pull("rlm/rag-prompt")
        retriever = vector_store.as_retriever()
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("RAG Chain Schema: ", rag_chain.input_schema.schema())

        rag_tool = rag_chain.as_tool(
            name="cbt_knowledge_expert",
            description="Generates concise answers to questions about cognitive behavioral therapy.",
        )

        return rag_tool

    # helpers
    def list_collections(self):
        existing_collections = self.client.list_collections()
        return existing_collections


# ChromaDB

# "client": chroma DB 전체 관리
# "collection": table - dataset 종류 (cbt, 심리상담, qna)

# cbt collection


# TODO: Move this to a utils file
def str_to_document(text: str):
    # Split the string into page_content and metadata
    page_content_part, metadata_part = text.split(" metadata=")

    # Extract page content
    page_content = page_content_part.split("page_content=", 1)[1].strip("'")

    # parse metadata string to dictionary
    metadata = ast.literal_eval(metadata_part)

    return Document(page_content=page_content, metadata=metadata)


# TODO: Move this to a utils file
def document_to_str(doc: Document):
    return f"page_content='{doc.page_content}' metadata={doc.metadata}"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
