from langchain_chroma import Chroma
from ingest import get_embedding_function
import config


def load_index():
    embed_fn = get_embedding_function()
    vectordb = Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embed_fn,
    )
    return vectordb


def retrieve(vectordb, question, k=3):
    results = vectordb.similarity_search_with_relevance_scores(question, k=k)
    return results
