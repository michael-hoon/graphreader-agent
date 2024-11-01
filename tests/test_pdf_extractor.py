import pytest
from kg_builder.pdf_extractor import PDFExtractor

@pytest.fixture
def pdf_extractor():
    return PDFExtractor()

def test_pdf_extraction(pdf_extractor):
    pdf_texts = pdf_extractor.extract_texts()
    
    # assertions
    assert isinstance(pdf_texts, dict)
    assert len(pdf_texts) > 0  # at least one document was loaded
    for doc_name, text in pdf_texts.items():
        assert isinstance(doc_name, str)
        assert isinstance(text, str)
        assert len(text) > 0  # text should not be empty, otherwise extraction failed
