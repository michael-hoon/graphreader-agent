import asyncio
from kg_builder.pdf_extractor import PDFExtractor
from kg_builder.kg_processor import KGProcessor

async def main():
    extractor = PDFExtractor(docs_path="docs/")

    print("Extracting text from PDFs...")
    extracted_texts = extractor.extract_texts()
    if not extracted_texts:
        print("No documents found in 'docs' folder. Please add PDF files and try again.")
        return

    processor = KGProcessor(chunk_size=1000, chunk_overlap=200)

    # process each document and upload to Neo4j
    for document_name, text in extracted_texts.items():
        print(f"Processing document: {document_name}")
        await processor.process_document(text, document_name=document_name)
    
    print("KG construction and upload completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
