import os
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

# custom library
from CbtRAG import CbtRAG, AgentGraph

# command line interface
import click

# evaluation library (DeepEval)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)

# to activate neo4j debug, uncomment codes below
# from neo4j.debug import watch
# watch("neo4j")

# to load the .env file
load_dotenv()

def create_index_cli(cbt_rag: CbtRAG):
    """
    CLI to help user create indexings for configured datasets
    """
    click.echo("==== Here are datasets ready for Indexing ====")
    datasets = cbt_rag.get_datasets()
    for dataset in datasets:
        val = click.prompt(
            f"Create indexing for dataset: {dataset}? (y/n)", type=str, default="n"
        )
        if val == "y":
            cbt_rag.create_indexing(dataset_name=dataset)
            click.echo(f"Indexing for dataset: {dataset} created successfully!")

def print_query_config_cli(cbt_rag: CbtRAG):
    """
    CLI to print query configurations
    """
    # 2. Context Retrieval (eval)
    click.echo("Ready to query with following options!")
    # pre-retrieval options
    click.echo("    Pre-Retrieval:")
    multi_query_expansion = cbt_rag.get_multi_query_expansion()
    click.echo(f"       Multi Query Expansion: {multi_query_expansion}")

    # post-retrieval options
    click.echo("    Post-Retrieval:")
    context_top_k = cbt_rag.get_context_top_k()
    context_reranking = cbt_rag.get_context_reranking()
    context_compression = cbt_rag.get_context_compression()

    click.echo(f"       Context Top K: {context_top_k}")
    click.echo(f"       Context Reranking: {context_reranking}")
    click.echo(f"       Context Compression: {context_compression}")

    # query db options
    click.echo("    Indexing:")
    datasets = cbt_rag.get_retrieval_datasets()
    for dataset in datasets:
        click.echo(
            f"       - index: {dataset["dataset_name"]}, tool: {dataset["tool_name"]}"
        )

def get_evaluation_dataset():
    """
    Get evaluation dataset
    """
    # read csv file: /data/evaluation/ground_truth.csv
    df = pd.read_csv("./data/evaluation/ground_truth.csv")
    # store in list of dictionaries
    # keys: input("User_Question"), expected_output("Best_Response")
    test_cases = []
    for i, row in df.iterrows():
        test_cases.append(
            {"input": row["User_Question"], "expected_output": row["Best_Response"]}
        )

    return test_cases

def eval_graph(graph: AgentGraph):
    click.echo("Evaluating RAG Chatbot...")
    # define metrics
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()
    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    # create a evaluation dataset
    eval_dataset: list[dict] = get_evaluation_dataset()
    # iterate through evaluation dataset
    test_cases = []
    for raw_test_case in eval_dataset:
        # create test_cases by using output from `graph.query()`
        output = graph.query(raw_test_case["input"])
        actual_output = output.get("response")

        retrieved_docs = []

        for sample_response in output["sample_responses"]:
            retrieved_docs.append(sample_response.page_content)

        for cbt_context in output["cbt_contexts"]:
            retrieved_docs.append(cbt_context.page_content)

        for socratic_context in output["socratic_contexts"]:
            retrieved_docs.append(socratic_context.page_content)

        # print("Retrieved Docs:")
        # pretty_print_docs(retrieved_docs_in_str)

        test_case = LLMTestCase(
            input=raw_test_case["input"],
            actual_output=actual_output,
            expected_output=raw_test_case["expected_output"],
            retrieval_context=retrieved_docs,
        )
    test_cases.append(test_case)

    eval_result = pd.DataFrame()
    # columns: input, expected_output, actual_output, contextual_precision, contextual_recall, contextual_relevancy, answer_relevancy, faithfulness

    for test_case in test_cases:
        print_divider()

        print(" Input: ", test_case.input)
        print(" Expected Output: ", test_case.expected_output)
        print(" Actual Output: ", test_case.actual_output)
        # print("Retrieval Context: ", test_case.retrieval_context
        # calculate metrics
        contextual_precision.measure(test_case)
        print(" Contextual Precision Score: ", contextual_precision.score)
        # print(" Contextual Precision Reason: ", contextual_precision.reason
        contextual_recall.measure(test_case)
        print(" Contextual Recall Score: ", contextual_recall.score)
        # print(" Contextual Recall Reason: ", contextual_recall.reason
        contextual_relevancy.measure(test_case)
        print(" Contextual Relevancy Score: ", contextual_relevancy.score)
        # print(" Contextual Relevancy Reason: ", contextual_relevancy.reason
        answer_relevancy.measure(test_case)
        print(" Answer Relevancy Score: ", answer_relevancy.score)
        # print(" Answer Relevancy Reason: ", answer_relevancy.reason
        faithfulness.measure(test_case)
        print(" Faithfulness Score: ", faithfulness.score)
        # print(" Faithfulness Reason: ", faithfulness.reason)

    new_row = pd.DataFrame(
        {
            "input": [test_case.input],
            "expected_output": [test_case.expected_output],
            "actual_output": [test_case.actual_output],
            "contextual_precision": [contextual_precision.score],
            "contextual_recall": [contextual_recall.score],
            "contextual_relevancy": [contextual_relevancy.score],
            "answer_relevancy": [answer_relevancy.score],
             "faithfulness": [faithfulness.score],
        }
    )

    eval_result = pd.concat([eval_result, new_row], ignore_index=True)

    return eval_result

def visualize_eval_result():
    eval_result = pd.read_csv("./data/evaluation/eval_result.csv")
    columns_to_display = [
        "input",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "answer_relevancy",
        "faithfulness",
    ]

    # Create a new DataFrame with only the columns we want to display
    display_df = eval_result[columns_to_display]
    metric_columns = [
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "answer_relevancy",
        "faithfulness",
    ]
    display_df[metric_columns] = display_df[metric_columns].round(2)
    # Create the table
    table = tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)

    # Print the table
    print("Evaluation Results:")
    print(table)

    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    summary_stats = (
        display_df[metric_columns].agg(["mean", "median", "min", "max"]).round(2)
    )
    summary_table = tabulate(summary_stats, headers="keys", tablefmt="grid")
    print(summary_table)

    print_divider()

# mode (required): 'eval' for evaluation, 'chat' for chat mode
@click.command()
@click.option("--eval", "mode", flag_value="eval", help="Run in evaluation mode.")
@click.option("--chat", "mode", flag_value="chat", help="Run in chat mode.")
def cli(mode):
    """
    Command Line Interface for CBT-RAG Chatbot

    @args
    - mode: 'eval' for evaluation which returns evaluation metrics, 'chat' for interactive chat mode
    """
    if mode != "eval" and mode != "chat":
        raise ValueError("Please specify mode as 'eval' or 'chat'")

    click.echo("Welcome to CBT-RAG chatbot ✨")
    print_divider()
    config = "./config.yaml"
    cbt_rag = CbtRAG(config=config)

    # 1. Create Indexing
    click.echo("")
    createIndex = click.prompt(
        "Do you want to create indexing? (y/n)", type=str, default="n"
    )
    if createIndex == "y":
        create_index_cli(cbt_rag)

    # 2. Show Query Configurations
    click.echo("")
    print_divider()
    print_query_config_cli(cbt_rag)
    print_divider()

    # 3. Create RAG Architecture (Agent Graph)
    click.echo("Creating RAG Chatbot...")
    graph = AgentGraph(cbt_rag)

    if mode == "eval":
        startEval = click.prompt(
            "Do you want to start evaluation? (y/n)", type=str, default="n"
        )
        if startEval == "y":
            click.echo("Evaluating RAG Chatbot...")

            # get evaluate result
            eval_result = eval_graph(graph)

            print_divider()

            # save eval result as csv file
            eval_result.to_csv("./data/evaluation/eval_result.csv", index=False)

            click.echo(
                "Evaluation completed! Results saved in ./data/evaluation/eval_result.csv"
            )

    elif mode == "chat":
        click.echo("Not supported yet!")

    print_divider()
    click.echo("Printing out evaluation results...")
    # 4. printout metrics from /data/evaluation/eval_result.csv

    visualize_eval_result()

if __name__ == "__main__":
    # .env file should contain Google Generative AI API key
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in .env file")

    cli()  # pylint: disable=no-value-for-parameter


# Helper Functions -----------------------------------------------------------------------
def print_divider():
    click.echo(
        "------------------------------------------------------------------------------------------------------------------"
    )

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )