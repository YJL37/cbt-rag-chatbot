import os
# from CbtRAG import CbtRAG
from dotenv import load_dotenv

# command line interface
import click 

# python run.py --eval
# python run.py --chat

load_dotenv()


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# mode (required): 'eval' for evaluation, 'chat' for chat mode
@click.command()
@click.option('--eval', 'mode', flag_value='eval', help="Run in evaluation mode.")
@click.option('--chat', 'mode', flag_value='chat', help="Run in chat mode.")
def cli(mode):
    if mode != "eval" and mode != "chat":
        raise ValueError("Please specify mode as 'eval' or 'chat'")

    click.echo("Welcome to CBT-RAG chatbot ✨")

    # eval mode
    # chat mode


if __name__ == "__main__":
    # .env file should contain Google Generative AI API key
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in .env file")


    cli()

    # cbt_rag = CbtRAG()

    # cbt_rag.create_indexing(dataset_name = "cbt_collection",
    #                         path_names = ["./data/cbt_and_techniques/Cognitive_Behavioral_Therapy_Strategies.pdf",
    #                                    "./data/cbt_and_techniques/therapists_guide_to_brief_cbtmanual.pdf",
    #                                    "./data/cbt_and_techniques/wellbeing-team-cbt-workshop-booklet-2016.pdf"]
    #                         )

    # sample_query = "What is CBT?"
    # contexts = cbt_rag.retrieve_contexts(
    #     dataset_name="cbt_collection", query = sample_query
    # )
    # print("Retrieved Contexts: " )
    # pretty_print_docs(contexts)

    # cbt_rag.query_with_tools(sample_query, "cbt_collection")
