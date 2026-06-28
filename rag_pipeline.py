import os
import re
import numpy as np
from typing import List, Dict, Any
import ollama

class RAGPipeline:
    def __init__(self, doc_dir: str = "documents", model: str = "qwen2.5:0.5b"):
        self.doc_dir = doc_dir
        self.model = model
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_documents()
        self.build_tfidf_index()

    def load_documents(self):
        if not os.path.exists(self.doc_dir):
            print(f"Warning: Document directory '{self.doc_dir}' does not exist.")
            return

        for filename in os.listdir(self.doc_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.doc_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split by double newlines to get paragraphs/sections
                paragraphs = content.split("\n\n")
                for i, para in enumerate(paragraphs):
                    para = para.strip()
                    if len(para) > 10:  # Ignore trivial empty lines
                        self.chunks.append({
                            "text": para,
                            "source": filename,
                            "chunk_id": f"{filename}_{i}",
                            "embedding": None
                        })
        print(f"Loaded {len(self.chunks)} chunks from documents.")

    def build_tfidf_index(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [c["text"] for c in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            print("TF-IDF fallback index built successfully.")
        except Exception as e:
            print(f"Failed to build TF-IDF index: {e}")

    def get_ollama_embedding(self, text: str) -> List[float]:
        try:
            # Generate embedding using the specified model
            res = ollama.embeddings(model=self.model, prompt=text)
            return res["embedding"]
        except Exception as e:
            print(f"Ollama embedding failed for text: {text[:30]}... Error: {e}")
            raise e

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        # Try Ollama embeddings first
        try:
            query_embedding = self.get_ollama_embedding(query)
            # Compute embeddings for chunks if not already done
            for chunk in self.chunks:
                if chunk["embedding"] is None:
                    chunk["embedding"] = self.get_ollama_embedding(chunk["text"])
            
            # Compute cosine similarity
            similarities = []
            q_vec = np.array(query_embedding)
            for chunk in self.chunks:
                c_vec = np.array(chunk["embedding"])
                sim = np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
                similarities.append(sim)
            
            # Sort by similarity
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(similarities[idx]),
                    "method": "ollama_embeddings"
                })
            return results
        except Exception as e:
            print(f"Retrieval falling back to TF-IDF due to: {e}")
            return self.retrieve_tfidf(query, top_k)

    def retrieve_tfidf(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if self.vectorizer is None or self.tfidf_matrix is None:
            return self.retrieve_keyword(query, top_k)
        
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": float(similarities[idx]),
                "method": "tf_idf"
            })
        return results

    def retrieve_keyword(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words.intersection(chunk_words))
            scores.append(overlap)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": float(scores[idx]),
                "method": "keyword_overlap"
            })
        return results
