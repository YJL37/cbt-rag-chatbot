import pandas as pd
from langchain.docstore.document import Document


class ScriptCSVManager:
    """
    - load csv
    - process csv to documents
    """

    # path_name => chunks
    def __init__(self, path_name):
        self.path_name = path_name

    def load_csv(self):
        df = pd.read_csv(self.path_name)

        df = df.drop_duplicates(subset=["Context", "Response"])

        # rename "context" column to "user_question" for better readability
        df = df.rename(columns={"Context": "user_question"})
        df = df.rename(columns={"Response": "response"})
        # print(df[pd.to_numeric(df["response"], errors="coerce").isna()])

        df["response"] = df["response"].astype(str)

        # Clean the user_question column
        df["user_question_clean"] = df["user_question"].str.strip().str.lower()

        # Sort the dataframe to ensure consistent results
        df_sorted = df.sort_values("user_question_clean")

        # Remove duplicates, keeping the first occurrence of each user_question
        df_unique = df_sorted.drop_duplicates(
            subset=["user_question_clean"], keep="first"
        )

        # Combine responses for the same user_question
        df_combined = (
            df_unique.groupby("user_question_clean")
            .agg(
                {
                    "user_question": "first",
                    "response": lambda x: "---".join(dict.fromkeys(x)),
                }
            )
            .reset_index(drop=True)
        )

        # Final check for any remaining duplicates
        df_final = df_combined.drop_duplicates(subset=["response"])

        return df_final

    def process_csv(self, df):
        """
        process df into text documents

        @return docs: list of documents (langchain Document object)
        """

        # process df to list of documents
        documents = []

        for index, row in df.iterrows():
            documents.append(
                Document(
                    page_content=row["user_question"],
                    metadata={"row_id": index, "response": row["response"]},
                )
            )

        return documents
