import yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_yaml_config(config_file_path):
    """Loads YAML configuration from a file, handling potential errors.

    @args:
      config_file_path: The path to the YAML configuration file.

    @returns:
      The parsed YAML data as a Python dictionary, or None if an error occurs.
    """

    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            return config_data
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        exit(1)

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)


class Config:
    """
    RAG Configuration manager class
    -  please use getter functions to access configuration values

    @args
    - config: processed .yaml file with load_yaml_config() file
    """

    def __init__(self, config):
        self.config = load_yaml_config(config)

    def get_dataset_names(self):
        return self.config["datasets"].keys()

    def get_dataset_info(self, dataset_name):
        return self.config["datasets"][dataset_name]

    def get_dataset_type(self, dataset_name):
        return self.config["datasets"][dataset_name]["database_type"]

    def get_dataset_description(self, dataset_name):
        return self.config["datasets"][dataset_name]["description"]

    def get_dataset_embedding_model(self, dataset_name):
        model = self.config["datasets"][dataset_name]["embedding_model"]
        embedding_model = GoogleGenerativeAIEmbeddings(model=model)

        return embedding_model

    def get_dataset_files(self, dataset_name):
        """
        Get dataset files from configuration
        """
        return self.config["datasets"][dataset_name]["files"]
