import re
import ast

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# TODO: Move this to a utils file
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r"\s+", " ", text)
    return text


# TODO: Move this to a utils file
def str_to_document(text: str):
    # Split the string into page_content and metadata
    page_content_part, metadata_part = text.split(" metadata=")

    # Extract page content
    page_content = page_content_part.split("page_content=", 1)[1].strip("'")

    # parse metadata string to dictionary
    metadata = ast.literal_eval(metadata_part)

    return Document(page_content=page_content, metadata=metadata)


class PDFManager:
    """
    - load document
    - preprocess document
    - split documents into chunks
    """

    # path_name => chunks
    def __init__(self, path_name):
        self.path_name = path_name

    def load_pdf(self):
        """
        load pdf documents

        @return pages: splitted pdf pages
        """
        loader = PyPDFLoader(self.path_name)
        pages = loader.load_and_split()

        return pages

    def process_pdf(self, pages):
        """
        process pdf documents into text chunks

        @return docs: list of documents (langchain Document object)
        """

        # split pdfs into texts

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-2",
            chunk_size=100,
            chunk_overlap=0,
        )

        raw_chunks = text_splitter.split_documents(pages)

        # Convert Document objects into strings
        chunks = [str(doc) for doc in raw_chunks]
        # Preprocess the text
        chunks = [preprocess_text(chunk) for chunk in chunks]
        # convert strings to Document objects
        docs = [str_to_document(chunk) for chunk in chunks]

        print("    Number of splitted tokens:", len(docs))

        return docs
