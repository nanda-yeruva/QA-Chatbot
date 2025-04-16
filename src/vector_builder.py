from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import pickle
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

current_file_dir = os.path.dirname(__file__)
project_root_dir = os.path.dirname(current_file_dir)

def count_terms_in_text(text, terms):
    """Count occurrences of specific terms in a given text."""
    term_counts = {term: 0 for term in terms}
    sorted_terms = sorted(terms, key=len, reverse=True)

    for term in sorted_terms:
        pattern = r'\b' + re.escape(term) + r'\b' if len(term.split()) == 1 else re.escape(term)
        matches = re.findall(pattern, text, re.IGNORECASE)
        term_counts[term] = len(matches)

    return term_counts

def compute_weighted_embedding(term_counts, embeddings, term_index_map):
    """Compute a weighted average embedding based on term frequencies."""
    embedding_dim = embeddings.shape[1]
    weighted_embedding = np.zeros(embedding_dim)
    total_weight = 0.0

    for term, count in term_counts.items():
        if count > 0:
            index = term_index_map[term]
            weighted_embedding += count * embeddings[index]
            total_weight += count

    return weighted_embedding / total_weight if total_weight > 0 else np.zeros(embedding_dim)

class MedicalEmbeddings:
    """Custom embedding class for medical terms using pre-trained clinical embeddings."""

    def __init__(self):
        try:
            self.knowledge_graph = pd.read_csv(os.path.join(project_root_dir, 'reference_docs', 'new_node_map_df.csv'))
            self.term_embeddings = np.load(os.path.join(project_root_dir, 'reference_docs', 'full_h_embed_hms.npy'))
        except FileNotFoundError as e:
            logging.error(f"Required file not found: {e}")
            raise

        self.terms = self.knowledge_graph.node_name.str.lower().values
        self.term_to_index = dict(zip(self.terms, self.knowledge_graph['global_graph_index'].values))

    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        return [self.embed_query(text) for text in tqdm(texts)]

    def embed_query(self, text):
        """Generate an embedding for a single query."""
        term_counts = count_terms_in_text(text, self.terms)
        return compute_weighted_embedding(term_counts, self.term_embeddings, self.term_to_index)

def main():
    """Main function to create vector databases."""
    pdf_path = os.path.join(project_root_dir, 'reference_docs', 'HER2_paper_stripped.pdf')
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at {pdf_path}")
        return

    logging.info("Loading PDF document...")
    pdf_loader = PyPDFLoader(pdf_path)
    document_data = pdf_loader.load()

    logging.info("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = text_splitter.split_documents(document_data)

    vecdb_path = os.path.join(project_root_dir, 'vecdb')
    if os.path.exists(vecdb_path):
        shutil.rmtree(vecdb_path)

    logging.info("Building vector database...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_database = Chroma.from_documents(
        document_chunks,
        collection_name="RAG_vector_db",
        embedding=embedding_model,
        persist_directory=vecdb_path
    )
    vector_database.persist()
    with open(os.path.join(vecdb_path, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(embedding_model, f)

    logging.info("Building clinical vector database...")
    medical_embeddings = MedicalEmbeddings()
    with open(os.path.join(vecdb_path, 'clinical_embeddings_model.pkl'), 'wb') as f:
        pickle.dump(medical_embeddings, f)

    clinical_vector_database = Chroma.from_documents(
        document_chunks,
        collection_name="clinical_vector_db",
        embedding=medical_embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=vecdb_path
    )
    clinical_vector_database.persist()

if __name__ == '__main__':
    main()