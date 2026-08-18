from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config


def load_pdfs(data_dir: Path):
    pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        raw_pages = loader.load()
        for i, page in enumerate(raw_pages):
            page.metadata["document_name"] = pdf_path.stem
            page.metadata["page_number"] = i + 1
        pages.extend(raw_pages)
        print(f"  -> {len(raw_pages)} pages loaded")
    return pages


def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('document_name', 'unknown')}_chunk_{idx}"
    return chunks


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def build_index(chunks):
    embed_fn = get_embedding_function()
    print(f"Embedding {len(chunks)} chunks using 'local' provider ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_fn,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"Done. Index saved to {config.CHROMA_DIR}/")
    return vectordb


if __name__ == "__main__":
    pages = load_pdfs(config.DATA_DIR)
    print(f"Loaded {len(pages)} pages.")
    chunks = chunk_documents(pages)
    print(f"Created {len(chunks)} chunks.")
    vectordb = build_index(chunks)
    print("Vector index built and persisted.")
