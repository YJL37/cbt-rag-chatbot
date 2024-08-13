import sys

from CbtRAG.pdf_manager import PDFManager
from CbtRAG.vector_db_manager import VectorDBManager

from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from langgraph.prebuilt import create_react_agent

class CbtRAG:
    """
    @args
    - path_names: dataset paths
    """
    def __init__(self):
        # LLM
        # temperature = llm이 얼마나 창의적이냐 (0 ~ 1, 0.7)
        # llm: llm for generate multiple queries
        self.llm = GoogleGenerativeAI(model = "gemini-1.5-flash", temperature=0.7)
        # chat_llm: llm for main chatbot
        self.chat_llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature=0)

        # vectorDBManager: vectorDB Table들을 관리
        self.vector_db_manager = VectorDBManager()

        # self.graph_manager = GraphManager()
        
    
    def create_indexing(self, dataset_name, path_names): 
        print(f"==== Create Indexing '{dataset_name}' ====")
        # Load Document
        # print(load_pdf(self.path_name)[0])

        # create vector db
        # upload docs to vector db
        # collection_name = "cbt_collection"
        self.vector_db_manager.init_db(collection_name = dataset_name)

        docs = []
        for path_name in path_names:
            pdf_manager = PDFManager(path_name=path_name)
            # load pdf 
            pages = pdf_manager.load_pdf()
            # process pdf
            docs.extend(pdf_manager.process_pdf(pages))

        self.vector_db_manager.upload_docs(docs=docs, collection_name=dataset_name)
        
        # VectorDBManager.upload_docs(docs)
        # VectorDBManager.retrieve_context(query)

        # create knowledge graph
        # upload docs to graph
        # KnowledgeGraphManager
        # KnowledgeGraphManager.upload_docs(docs)
        # KnowledgeGraphManager.retrieve_context(query)

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

        retrieved_contexts = self.vector_db_manager.retrieve_contexts_with_multi_queries(
            query = query, 
            collection_name = dataset_name,
            llm = self.llm,
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
        tool = self.vector_db_manager.create_rag_chain_tool_with_collection(dataset_name, self.chat_llm)
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

