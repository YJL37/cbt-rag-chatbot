import sys

from .script_csv_manager import ScriptCSVManager
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

    # GETTERS ---------------------------------------------------------------------------
    def get_vector_db_manager(self):
        return self.vector_db_manager

    def get_chat_llm(self):
        return self.chat_llm

    # RAG: Indexing ---------------------------------------------------------------------
    def create_indexing(self, dataset_name):
        """
        function to create indexing

        @args
        - dataset_name
        """

        dataset_type = self.config.get_dataset_type(dataset_name)
        files = self.config.get_dataset_files(dataset_name)

        print(
            f"==== Create Indexing '{dataset_name}' at '{dataset_type} database' ===="
        )

        docs = []

        # process pdf files into manageable documents (text tokens)
        if files[0]["type"] == "pdf":
            for file in files:
                print("    Processing file: " + file["path"])
                pdf_manager = PDFManager(path_name=file["path"])
                # load pdf
                pages = pdf_manager.load_pdf()
                # process pdf
                docs.extend(pdf_manager.process_pdf(pages))
        # process cvs files into manageable documents (text tokens)
        elif files[0]["type"] == "csv":
            # only one file accepted for csv
            for file in files:
                print("    Processing file: " + file["path"])
                script_csv_manager = ScriptCSVManager(path_name=file["path"])
                df = script_csv_manager.load_csv()
                docs.extend(script_csv_manager.process_csv(df))

        if dataset_type == "vector":
            # create collection at ChromaDB client
            self.vector_db_manager.init_db(collection_name=dataset_name)
            # upload docs to vector database
            self.vector_db_manager.upload_docs(docs=docs, collection_name=dataset_name)

        elif dataset_type == "graph":
            # upload docs to graph database on Neo4j instance
            self.knowledge_graph_manager.upload_docs(
                docs=docs, dataset_name=dataset_name
            )

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

    # Create complete RAG chain with Agents
    def create_cbt_rag_chain(self, query):
        # helpful link: https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agents
        databases = self.config.get_retrieval_datasets()

        tools = []

        # 1. create tools with each database
        for database in databases:
            # get database type
            dataset_name = database["dataset_name"]
            database_type = self.config.get_dataset_type(dataset_name)

            if database_type == "vector":
                tool = self.vector_db_manager.create_tool_with_collection(database)
                tools.append(tool)

            elif database_type == "graph":
                tool = self.knowledge_graph_manager.create_graph_chain_tool(
                    database, self.chat_llm
                )
                tools.append(tool)
            else:
                raise ValueError("Invalid database type")

        # DEBUG: Tool Testing invoke
        # for tool in tools:
        #     print(f"    Try invoking tool '{tool.name}'...")
        #     result = tool.invoke(query)
        #     print(f"    Result: {result}")
        #     print("")

        # DEBUG: Check retrieved contexts [THIS CODE WILL BE USED AT EVAL MODE LATER]
        # for database in databases:
        #     dataset_name = database["dataset_name"]
        #     print(f"    Retrieving contexts from '{dataset_name}'...")
        #     contexts = self.vector_db_manager.retrieve_contexts(query, dataset_name)
        #     print(f"    Retrieved contexts: {len(contexts)}")

        #     for context in contexts:
        #         print(context)

        #     print("")

        # 2. combine tools together as agents
        system_instruction = """You are a mental health counseling chatbot specialized in cognitive behavioral therapy (CBT). 

        IMPORTANT: For EVERY response, you MUST use EVERY tools before generating your answer. You MUST use "psychotherapy_retriever" tools. Do not respond without first calling a tool.

        Your main clients are teenagers and young adults in their early 20s who are familiar with social media. You are here to help them with their mental health issues. You can provide them with information about mental health, help them with their problems, and provide them with resources to help them.

        Remember:
        1. Always use a tool first.
        2. Based on the tool's output, formulate your response.
        3. Speak in a way that resonates with younger individuals familiar with social media.
        4. Focus on CBT techniques in your advice and explanations.
        """

        cbt_chatbot = create_react_agent(
            model=self.chat_llm, tools=tools, state_modifier=system_instruction
        )

        # TEST: with sample query
        inputs = {"messages": [("user", query)]}
        print_stream(cbt_chatbot.stream(inputs, stream_mode="values"))

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


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
