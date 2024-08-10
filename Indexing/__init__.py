from Indexing.pdf_manager import PDFManager
from Indexing.vector_db_manager import VectorDBManager

class Indexing:
    """
    @args
    - path_names: dataset paths
    """
    def __init__(self, path_names):
        self.path_names = path_names
        
    
    def test(self): 
        # Load Document
        # print(load_pdf(self.path_name)[0])

        # create vector db
        # upload docs to vector db
        vector_db_manager = VectorDBManager()
        collection_name = "cbt_collection"
        vector_db_manager.init_db(collection_name = collection_name)

        for path_name in self.path_names:
            pdf_manager = PDFManager(path_name=path_name)
            # load pdf 
            pages = pdf_manager.load_pdf()
            # process pdf
            docs = pdf_manager.process_pdf(pages)

            vector_db_manager.upload_docs(docs=docs, collection_name=collection_name)

        # VectorDBManager.upload_docs(docs)
        # VectorDBManager.retrieve_context(query)

        # create knowledge graph
        # upload docs to graph
        # KnowledgeGraphManager
        # KnowledgeGraphManager.upload_docs(docs)
        # KnowledgeGraphManager.retrieve_context(query)
