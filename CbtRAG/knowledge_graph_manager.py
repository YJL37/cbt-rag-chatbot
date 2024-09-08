from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from langchain.chains import GraphCypherQAChain
from langchain_core.prompts.prompt import PromptTemplate

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Type


class ChainInput(BaseModel):
    query: str = Field(description="The query to ask the graph database")


class GraphChainTool(BaseTool):
    name: str
    description: str
    chain: GraphCypherQAChain
    args_schema: Type[BaseModel] = ChainInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run the tool with the given query."""
        # when AI uses GraphChainTool with query as input, below code will be executed
        return self.chain.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Run the tool asynchronously with the given query."""
        raise NotImplementedError("GraphChainTool does not support async")

class KnowledgeGraphManager:
    def __init__(self):
        # Initialize knowledge graph connection, etc.
        self.graph = Neo4jGraph()
        self.graph_llm = ChatOpenAI(temperature = 0, model_name="gpt-3.5-turbo")
        pass

    # Add methods for knowledge graph operations

    def upload_docs(self, docs, dataset_name):
        """
        Upload docs to the graph
        """
        print("    Initializing database with graph: ", dataset_name)

        llm_transformer = LLMGraphTransformer(llm = self.graph_llm)

        graph_documents = llm_transformer.convert_to_graph_documents(docs)

        self.graph.add_graph_documents(graph_documents)
        print("    Done uploading documents to graph: ", dataset_name)

    
    def create_graph_chain_tool(self, database, llm):
        print("    Creating graph chain tool for: ", database["dataset_name"])

        CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.
        Schema:
        {schema}
        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.
        Examples: Here are a few examples of generated Cypher statements for particular questions:
        # How many people played in Top Gun?
        MATCH (m:Movie {{name:"Top Gun"}})<-[:ACTED_IN]-()
        RETURN count(*) AS numberOfActors

        The question is:
        {question}"""

        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
        )


        # graph database ---- query
        graph_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=self.graph_llm,
            qa_llm=llm,
            verbose=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
        )

        # response = graph_chain.invoke({"query": "How can I improve my self-esteem?"})
        # print("    Response: ", response)

        graph_tool = GraphChainTool(
            name=database["tool_name"],
            description=database["tool_description"],
            chain=graph_chain,
        )

        # print("    Graph Tool: ", graph_tool, graph_tool.args)
        # print("    Graph Chain Schema: ", graph_tool.input_schema.schema())

        print("    Successfully created graph chain tool: ", database["tool_name"])

        return graph_tool
# 1. initialize client (ChromaDB, 여러개의 collection을 관리)
# Neo4j => community version (free) => 하나의 instance밖에 만들지 못한다...
# -> 하나의 graph database밖에 못 만든다. 