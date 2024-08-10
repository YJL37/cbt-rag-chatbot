import argparse
from Indexing import Indexing
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in .env file")
    # parser = argparse.ArgumentParser(description="Create and index a database")
    # parser.add_argument('dataset_path', type=str, help="Path to the dataset")
    # parser.add_argument('database_type', type=str, choices=['vector', 'graph'], help="Type of the database (vector or graph)")
    
    # args = parser.parse_args()

    indexing = Indexing(path_names = ["./data/cbt_and_techniques/Cognitive_Behavioral_Therapy_Strategies.pdf",
                                       "./data/cbt_and_techniques/therapists_guide_to_brief_cbtmanual.pdf",
                                       "./data/cbt_and_techniques/wellbeing-team-cbt-workshop-booklet-2016.pdf"])
    indexing.test()
    pass

    