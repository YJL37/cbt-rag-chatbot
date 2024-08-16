import sys

from .pdf_manager import PDFManager
from .vector_db_manager import VectorDBManager
from .knowledge_graph_manager import KnowledgeGraphManager
from .config import Config

from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from langgraph.prebuilt import create_react_agent


class CbtRAG:
    """
    @args
    - path_names: dataset paths
    """

    def __init__(self, config):
        self.config = Config(config)

        # LLM
        # temperature = llm이 얼마나 창의적이냐 (0 ~ 1, 0.7)
        # llm: llm for generate multiple queries
        self.llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        # chat_llm: llm for main chatbot
        self.chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        # initialize DB managers
        self.vector_db_manager = VectorDBManager()  # chromaDB tables
        self.knowledge_graph_manager = (
            KnowledgeGraphManager()
        )  # Neo4j instance (single graph db)

    def create_indexing(self, dataset_name):
        dataset_type = self.config.get_dataset_type(dataset_name)
        files = self.config.get_dataset_files(dataset_name)

        print(
            f"==== Create Indexing '{dataset_name}' at '{dataset_type} database' ===="
        )

        if dataset_type == "vector":
            # create collection at ChromaDB client
            self.vector_db_manager.init_db(collection_name=dataset_name)

            # process pdf files into manageable documents (text tokens)
            docs = []
            for file in files:
                pdf_manager = PDFManager(path_name=file["path"])
                # load pdf
                pages = pdf_manager.load_pdf()
                # process pdf
                docs.extend(pdf_manager.process_pdf(pages))

            # upload docs to vector database
            self.vector_db_manager.upload_docs(docs=docs, collection_name=dataset_name)

        elif dataset_type == "graph":
            pass

        else:
            raise ValueError("Invalid dataset type")

    def retrieve_contexts(self, dataset_name, query):
        """
        Retrieve contexts from database

        @args
        - dataset_name
        - query
        """
        print("==== Retrieve Contexts ==== ")
        # vector db
        # if database type is vector:
        # retrieved_contexts = self.vector_db_manager.retrieve_contexts(
        #     query = query,
        #     collection_name = dataset_name,
        # )

        retrieved_contexts = (
            self.vector_db_manager.retrieve_contexts_with_multi_queries(
                query=query,
                collection_name=dataset_name,
                llm=self.llm,
            )
        )

        return retrieved_contexts

        # elif database type is graph:
        # self.graph_manager.retrieve_contexts(

        # )

    # TEST
    def test_query(self, query):
        template = """Question: {question}

        Answer: Let's think step by step."""
        prompt = PromptTemplate.from_template(template)

        chain = prompt | self.llm

        for chunk in chain.stream({"question": query}):
            sys.stdout.write(chunk)
            sys.stdout.flush()

    # TODO change dataset_name to list
    def query_with_tools(self, query, dataset_name):
        # create tool
        tool = self.vector_db_manager.create_rag_chain_tool_with_collection(
            dataset_name, self.chat_llm
        )
        tools = [tool]

        messages = [
            ("human", query),
        ]

        agent = create_react_agent(self.chat_llm, tools)

        for chunk in agent.stream({"messages": messages}, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

        # self.chat_llm.bind_tools(tools)

        # print(self.chat_llm.invoke(query))

    # helpers
    def list_vector_db_collections(self):
        return self.vector_db_manager.list_collections()

    # helpers for CLI
    def get_datasets(self):
        return self.config.get_dataset_names()

    def get_multi_query_expansion(self):
        return self.config.get_multi_query_expansion()

    def get_context_top_k(self):
        return self.config.get_context_top_k()

    def get_context_reranking(self):
        return self.config.get_context_reranking()

    def get_context_compression(self):
        return self.config.get_context_compression()

    def get_retrieval_datasets(self):
        return self.config.get_retrieval_datasets()
