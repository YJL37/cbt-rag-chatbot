# pdf or csv

from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path_name):
    """
    @desc Load PDF Document
    @params path_name: file path
    @return pages: []
    """
    
    loader = PyPDFLoader(path_name)
    pages = loader.load_and_split()

    return pages