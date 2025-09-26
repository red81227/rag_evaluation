import os
from sentence_transformers import SentenceTransformer, util

from .utils import read_all_files_to_bytes

class RAGEvaluator:
    def __init__(self, rag_service, file_path: str):
        self.rag_service = rag_service
        self.files_folder_path = file_path + "/files/"
        self.qas_folder_path = file_path + "/qas/"

    def evaluate(self, k: int = 3) -> tuple[float, float]:
        """Evaluate the RAG service using test files and Q&A pairs."""
        # 將測試文件放入 RAG 系統
        self.put_test_files(self.files_folder_path)
        # 讀取所有 Q&A 檔案
        qas = read_all_files_to_bytes(folder_path = self.qas_folder_path)

        # 對每個 Q&A 檔案計算命中率和 MRR，然後取平均值
        total_hit_rate = 0
        total_mrr = 0
        for filename, qas in qas.items():
            hit_rate, mrr = self.get_qa_set_hit_rate_and_mrr(qas, k)
            total_hit_rate += hit_rate
            total_mrr += mrr
        return total_hit_rate / len(qas), total_mrr / len(qas)

    def put_test_files(self, folder_path: str)-> None:
        """Put test files into the vector store."""
        pdf_contents = read_all_files_to_bytes(folder_path)
        for filename, file in pdf_contents.items():
            self.rag_service.add_document_and_update_store(content_bytes = file, filename=filename)
    
    def retrieve_context(self, query: str, k:int)-> tuple[str, list]:
        """Retrieve information to help answer a query."""
        retrieved_docs = self.rag_service.vector_store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    def get_qa_set_hit_rate_and_mrr(self, qas: list[dict], k: int = 3) -> tuple[float, float]:
        """Calculate the hit rate and MRR for a set of Q&A pairs."""
        hit_count = 0
        total_mrr = 0
        for qa in qas:
            query = qa.get("question", "")
            golden_source = qa.get("golden_source","")
            serialized, retrieved_docs = self.retrieve_context(query, k)
            retrieved_txt = [re_doc.page_content for re_doc in retrieved_docs]
            hit_rates = self.calculate_cosine_similarity(golden_source, retrieved_txt)
            mrr = self.get_mrr(hit_rates)
            if mrr > 0:
                hit_count += 1
                total_mrr += mrr

        return hit_count/len(qas), total_mrr/len(qas)
    
    @staticmethod    
    def calculate_cosine_similarity(golden_source: str, retrieved_chunks: list[str]) -> list[float]:
        """
        計算黃金出處與每個檢索片段之間的餘弦相似度。

        Args:
            golden_source: 包含正確答案的黃金出處字串。
            retrieved_chunks: 檢索到的多個文本片段組成的列表。

        Returns:
            一個包含相似度分數的列表。
        """
        # 載入預訓練的多語言模型，以支援中文文本
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        golden_embedding = model.encode(golden_source, convert_to_tensor=True)
        chunk_embeddings = model.encode(retrieved_chunks, convert_to_tensor=True)
        cosine_scores = util.cos_sim(golden_embedding, chunk_embeddings).cpu().numpy()
        return cosine_scores.flatten().tolist()

    @staticmethod
    def get_mrr(cosine_similarity):
        """Calculate the Mean Reciprocal Rank (MRR) from the cosine similarity scores."""
        mrr = 0
        for idx, similarity in enumerate(cosine_similarity):
            if similarity > 0.8:
                mrr += 1 / (idx + 1)
        return mrr
