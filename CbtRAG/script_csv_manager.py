import pandas as pd
from langchain.docstore.document import Document

# ScriptCSVManager

# - load csv as dataframe
# - process csv to docs

class ScriptCSVManager:
    def __init__ (self, path_name):
        self.path_name = path_name
        
    def load_csv(self):
        data = pd.read_csv(self.path_name)

        # rename "context" column to "question"
        # rename "Response" column to "response"
        data = data.rename(columns = {'Context':'question'})
        data = data.rename(columns = {'Response':'response'})

        data["response"] = data["response"].astype(str)

        # 5개의 row -> 1개의 row
        # questions: "I'm going ..."
        # response: "If everyone .... Hello... First thin"

        df_grouped = (
            data.groupby("question")["response"]
            .apply(lambda x: "\n".join(x))
            .reset_index()
        )

        return df_grouped
    
    def process_csv(self, df):
        # df -> docs

        docs = [] # Document[]
        for index, row in df.iterrows():
            doc = Document(
                page_content = row["question"],
                metadata = {"response": row["response"]},
            )
            docs.append(doc)
        return docs

        # => query를 page_content로 한다.
        #     user's query => embedding => dataset에 있는 가장 가까운 embedding (raw data -> embedding)