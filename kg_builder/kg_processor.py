import asyncio
from hashlib import md5
from datetime import datetime
from .neo4j_uploader import Neo4jUploader
from .data_models import Extraction

from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class KGProcessor:
    def __init__(self, model=None, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.neo4j_uploader = Neo4jUploader()

        # model config
        model = model or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.structured_llm = model.with_structured_output(Extraction) # pydantic model for generating list of atomic facts

        # authors used prompt based extraction but structured output can be more reliable

        # prompt config
        construction_system = """
        You are now an intelligent research assistant tasked with meticulously extracting both key elements and atomic facts from a research paper.
        1. Key Elements: Critical entities such as terms, hypotheses, methodologies, models, datasets, findings, and results that are central to the paper's purpose.
        2. Atomic Facts: The smallest, indivisible factual units presented as concise sentences. These include propositions, statements on research objectives, theories, hypotheses, conclusions, methods, relationships, metrics, and significant findings, etc. Exclude generic references to background information.

        Requirements:
        #####
        1. Ensure that all identified key concepts are accurately reflected in corresponding atomic facts.
        2. You should extract key elements and atomic facts comprehensively, especially those that are important and potentially query-worthy and do not leave out details.
        3. Whenever applicable, replace vague terms (e.g., “this study”) with precise terms from the paper, such as specific hypotheses, models, or experimental names.
        4. Ensure that all key concepts and atomic facts you extract are presented in the same language as the original text.
        """

        # 2. Focus on extracting content from sections like “Abstract,” “Introduction,” “Methodology,” “Results,” and “Discussion,” and avoid information from “References” or “Appendices” unless it contains substantive findings.

        # can be added in the prompt after incorporating markdown style extraction with pymupdf4llm, text is then separated by headers and subheaders for structure

        construction_human = """Use the given format to extract information from the 
        following input: {input}"""

        self.construction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    construction_system,
                ),
                (
                    "human", 
                    construction_human,
                ),
            ]
        )

        self.construction_chain = self.construction_prompt | self.structured_llm

    def encode_md5(self, text):
        return md5(text.encode("utf-8")).hexdigest()
    
    async def process_document(self, text, document_name):
        start = datetime.now()
        print(f"Started extraction at: {start}")

        chunks = self.text_splitter.split_text(text)
        print(chunks)
        print(f"Text split into {len(chunks)} chunks.")

        tasks = [
            asyncio.create_task(self.construction_chain.ainvoke({"input":chunk_text}))
            for chunk_text in chunks
        ] # for each chunk, asynchronously send text to LLM for extraction of atomic facts and key elements

        results = await asyncio.gather(*tasks)
        print(f"Finished LLM extraction after: {datetime.now() - start}")

        docs = [result.model_dump() for result in results]
        for index, doc in enumerate(docs):
            doc['chunk_id'] = self.encode_md5(chunks[index]) # each chunk and atomic fact is assigned a uuid using md5 hash
            doc['chunk_text'] = chunks[index]
            doc['index'] = index
            for af in doc['atomic_facts']:
                af['id'] = self.encode_md5(af['atomic_fact'])

        # upload data (chunks/atomic facts/key elements) to neo4j
        self.neo4j_uploader.upload_data(docs, document_name)

        # create NEXT relationship between chunks
        self.neo4j_uploader.create_next_relationships(document_name)
        print(f"Processing completed in: {datetime.now() - start}")