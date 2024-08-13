import argparse
from CbtRAG import CbtRAG
import os
from dotenv import load_dotenv

load_dotenv()

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in .env file")
    # parser = argparse.ArgumentParser(description="Create and index a database")
    # parser.add_argument('dataset_path', type=str, help="Path to the dataset")
    # parser.add_argument('database_type', type=str, choices=['vector', 'graph'], help="Type of the database (vector or graph)")
    
    # args = parser.parse_args()

    cbt_rag = CbtRAG()
    
    # cbt_rag.create_indexing(dataset_name = "cbt_collection", 
    #                         path_names = ["./data/cbt_and_techniques/Cognitive_Behavioral_Therapy_Strategies.pdf",
    #                                    "./data/cbt_and_techniques/therapists_guide_to_brief_cbtmanual.pdf",
    #                                    "./data/cbt_and_techniques/wellbeing-team-cbt-workshop-booklet-2016.pdf"]
    #                         )
    

    sample_query = "What is CBT?"
    # contexts = cbt_rag.retrieve_contexts(
    #     dataset_name="cbt_collection", query = sample_query
    # )
    # print("Retrieved Contexts: " )
    # pretty_print_docs(contexts)

    cbt_rag.query_with_tools(sample_query, "cbt_collection")
    pass



    