import os
from typing import Dict

from langchain_community.document_loaders.s3_directory import S3DirectoryLoader

from dotenv import load_dotenv
load_dotenv()

class PDFExtractor:
    def __init__(self, prefix: str = '', use_ssl: bool = False):
        """
        Initializes the PDFExtractor with MinIO S3 configuration.
        
        :param bucket: Name of the S3 bucket.
        :param prefix: Folder path inside the bucket (optional).
        :param use_ssl: Whether to use SSL (HTTPS).
        """
        self.bucket = os.getenv('MINIO_BUCKET')
        self.prefix = prefix
        self.use_ssl = use_ssl
        self.endpoint = os.getenv('MINIO_ENDPOINT')
        self.access_key = os.getenv('MINIO_ACCESS_KEY')
        self.secret_key = os.getenv('MINIO_SECRET_KEY')
        
        if not all([self.endpoint, self.access_key, self.secret_key]):
            raise ValueError("Missing required MinIO environment variables. Check .env file.")

    def extract_texts(self) -> Dict[str, str]:
        """
        Extracts text from all PDF documents in the specified MinIO bucket and prefix.
        :return: A dictionary where each key is the document name, and each value is the combined extracted text of all pages.
        """
        loader = S3DirectoryLoader(
            bucket=self.bucket,
            prefix=self.prefix,
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key, 
            aws_secret_access_key=self.secret_key, 
            use_ssl=self.use_ssl
        )
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