import ast
import os
from typing import List, TypedDict, Optional
from .cbt_rag import CbtRAG
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate

# LangGraph
from langgraph.graph import StateGraph, END

# helpers ---------------------------------------------------------------------
def str_to_document_for_script(response: str, score: float):
    return Document(page_content=response, metadata={"score": score})

class GraphState(TypedDict):
    query: Optional[str] = None
    sample_responses: Optional[List[Document]] = None
    cbt_contexts: Optional[List[Document]] = None
    socratic_contexts: Optional[List[Document]] = None
    response: Optional[str] = None

    
class AgentGraph:
    def __init__(self, cbt_rag: CbtRAG):
        self.cbt_rag = cbt_rag
        self.vector_db_manager = cbt_rag.get_vector_db_manager()
        self.chat_llm = cbt_rag.get_chat_llm()

        # TODO: StateGraph initialization
        workflow = StateGraph(GraphState)

        # TODO: add nodes
        workflow.add_node("Sample Response Retriever", self.retrieve_sample_responses)
        workflow.add_node("Final Counseling", self.generate_final_response)
        # workflow.add_node("Psychotherapy", self.retrieve_psychotherapy_context)
        workflow.add_node("CBT", self.retrieve_cbt_contexts)
        workflow.add_node("Socratic", self.retrieve_socratic_contexts)


        # TODO: add edges
        # node a, b
        # workflow.add_edge(a, b)
        workflow.set_entry_point("Sample Response Retriever")
        # workflow.add_edge("Sample Response Retriever", "Psychotherapy")
        workflow.add_edge("Sample Response Retriever", "CBT")
        # workflow.add_edge("Sample Response Retriever", "Socratic")
        # workflow.add_edge("Psychotherapy", "Final Counseling")
        workflow.add_edge("CBT", "Socratic")
        workflow.add_edge("Socratic", "Final Counseling")
        workflow.add_edge("Final Counseling", END)

        app = workflow.compile()
        
        self.app = app
        # image_data = app.get_graph().draw_mermaid_png()

        # with open("agent_graph.png", "wb") as f:
        #     f.write(image_data)
        
        # os.system("open agent_graph.png")
    
    def query(self, query):
        """
        Query the agent graph with user input query
        """
        return self.app.invoke({"query": query})

    # Functions for nodes in LangGraph ----------------------------------------------
    def retrieve_sample_responses(self, state):
        """
        Retrieve sample responses 

        @args state: GraphState [query]
        """
        query = state.get("query")
        collection_name = "counseling_script_collection"
        
        raw_contexts = self.vector_db_manager.retrieve_contexts(query, collection_name)

        # <Document>
        # page_content: context (query)
        # metadata: score, response (sample response)
        # ->
        # page_content: response (sample_response)
        # metadata: score

        # metadata:response is what we want as "page_content" in sample_responses[]
        formatted_contexts = []
        for context in raw_contexts:
            if context.metadata.get("response"):
                formatted_contexts.append(
                    str_to_document_for_script(
                        response=context.metadata.get("response"),
                        score=context.metadata.get("score"),
                    )
                )

        return { "sample_responses": formatted_contexts}

    def retrieve_cbt_contexts(self, state):
        """
        @args state: GraphState [query, sample_responses]
        @return cbt_contexts
        """
        
        query = state.get("query")
        sample_responses = state.get("sample_responses")
        formatted_sample_responses = [str(doc.page_content) for doc in sample_responses]

        collection_name = "cbt_knowledge_collection"

        input = f"User's thought: '{query}'. Sample therapist responses: '{formatted_sample_responses}'."

        contexts = self.vector_db_manager.retrieve_contexts(query = input, collection_name = collection_name)

        return {"cbt_contexts": contexts}
    
    def retrieve_socratic_contexts(self, state):
        """
        @args state: GraphState [query, sample_responses]
        @return socratic_contexts
        """
        
        query = state.get("query")
        sample_responses = state.get("sample_responses")
        formatted_sample_responses = [str(doc.page_content) for doc in sample_responses]

        collection_name = "socratic_questioning_collection"

        input = f"User's thought: '{query}'. Sample therapist responses: '{formatted_sample_responses}'."

        contexts = self.vector_db_manager.retrieve_contexts(query = input, collection_name = collection_name)

        return {"socratic_contexts": contexts}

    def generate_final_response(self, state):
        """
        @args state: GraphState [query, sample_responses, cbt_contexts, socratic_contexts]
        @return response
        """

        query = state.get("query")

        sample_responses = state.get("sample_responses")
        cbt_contexts = state.get("cbt_contexts")
        socratic_contexts = state.get("socratic_contexts")

        # 1. define llm
        chat_llm = self.chat_llm

        # 2. prompt engineering
        system_message = (
            "You are a compassionate and knowledgeable CBT counselor. "
            "You use Cognitive Behavioral Therapy techniques, particularly Socratic questioning, "
            "to help users reflect on their thoughts and challenge cognitive distortions. "
            "You will also incorporate relevant CBT context where applicable."
        )

        # Creating the template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "{query}"),
                (
                    "assistant",
                    "Here are some thoughts to consider based on cognitive behavioral therapy: "
                    "{sample_responses}. Additionally, consider these contexts from CBT literature: "
                    "{cbt_contexts}. And here are some reflective Socratic questions to help you think more deeply: "
                    "{socratic_contexts}.",
                ),
            ]
        )

        # 3. Create the chain with prompt and LLM
        chain = prompt | chat_llm

        # 4. Invoke the chain with the state values
        cbt_contexts_str = "\n".join([doc.page_content for doc in cbt_contexts])
        socratic_questioning_contexts_str = "\n".join(
            [doc.page_content for doc in socratic_contexts]
        )
        final_response = chain.invoke(
            {
                "query": query,
                "sample_responses": sample_responses,
                "cbt_contexts": cbt_contexts_str,
                "socratic_contexts": socratic_questioning_contexts_str,
            }
        )

        return {"response": final_response.content}


# StateGraph: Graph 관리자 (GraphState)
# - state: GraphState
# - Graph를 만든다
#     => add_node(), set_entry_point()
# - Graph를 수정한다.
# ...

