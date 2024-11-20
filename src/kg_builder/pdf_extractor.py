from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from typing import List, Dict
import os

class PDFExtractor:
    def __init__(self, docs_path: str = "../../docs/"):
        self.docs_path = docs_path

    def extract_texts(self) -> Dict[str, str]:
        """
        Extracts text from all PDF documents in the specified folder.
        :return: A dictionary where each key is the document name, and each value is the combined extracted text of all pages.
        """
        loader = PyPDFDirectoryLoader(self.docs_path)
        documents = loader.load()
        
        extracted_texts = {}

        # loop through each document and extract text from all pages
        for doc in documents:
            source = os.path.basename(doc.metadata["source"])
            if source in extracted_texts:
                extracted_texts[source] += "\n" + doc.page_content
            else:
                extracted_texts[source] = doc.page_content

        return extracted_texts
