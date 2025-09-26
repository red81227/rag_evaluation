"""
This module contains the RAG service for handling vector database and QA chain.
"""
from datetime import datetime
import os
import shutil
from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import DirectoryLoader,, Docx2txtLoader, TextLoader, UnstructuredFileLoader# type: ignore
from langchain_core.documents import Document # type: ignore
from langchain_community.document_loaders import UnstructuredPDFLoader

from ems_llm.configs.project_setting import (
    rag_config, 
)
from ems_llm.configs.logger_setting import log


class RAGService:
    """Service for Retrieval-Augmented Generation, including document management."""

    def __init__(self):
        
        # 初始化嵌入模型
        self.embed_model = OpenAIEmbeddings(
            model=rag_config.embedding_model,
            openai_api_key=rag_config.openai_api_key
        )

        # 確保文件和向量儲存目錄存在
        os.makedirs(rag_config.docs_path, exist_ok=True)
        os.makedirs(rag_config.vector_store_path, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=10, 
            length_function=len, 
            add_start_index=True,
            strip_whitespace=True
        )
        self.vector_store = self._load_or_create_vector_store()

    def add_document_and_update_store(self, content_bytes: bytes, filename: str):
        """
        將新的文件內容儲存、進行向量化，並增量更新到現有的向量資料庫中。
        這個方法取代了舊的 save_validated_document。

        Args:
            content_bytes (bytes): 文件的二進位內容。
            filename (str): 要儲存的檔名。
        """
        # 步驟 1: 將原始文件儲存到 source 資料夾
        file_path = os.path.join(rag_config.docs_path, filename)
        try:
            with open(file_path, 'wb') as f:
                f.write(content_bytes)
            log.info(f"Saved new document file: {file_path}")
        except IOError as e:
            log.error(f"Failed to save document file {filename}: {e}")
            raise

        # 步驟 2: 根據文件類型選擇適當的載入器
        try:
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension == '.pdf':
                loader = UnstructuredPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                # 對於其他格式，嘗試作為文本文件處理
                log.warning(f"Unknown file type {file_extension}, treating as text file")
                loader = TextLoader(file_path, encoding='utf-8')
            
            docs = loader.load()
            
            if not docs:
                log.warning(f"Document {filename} could not be processed or is empty. Skipping vector update.")
                return
                
            # 更新文檔的元數據
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "created_at": datetime.utcnow().isoformat()
                })
            
            
                
        except Exception as e:
            log.error(f"Failed to load and process document {filename}: {e}")
            return

        # 步驟 2.1: 保存原始內容用於行號計算
        original_contents = {}  # 存儲每頁的原始內容
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get('page', i + 1)
            original_contents[page_num] = doc.page_content

        # 步驟 3: 分割文檔
        texts = self.text_splitter.split_documents(docs)

        if not texts:
            log.warning(f"Document {filename} produced no text chunks after splitting. Skipping vector update.")
            return
        
         # 步驟 3.1: 為每個文本塊添加精確的位置信息
        for i, text_chunk in enumerate(texts):
            page_num = text_chunk.metadata.get('page', 1)
            start_index = text_chunk.metadata.get('start_index', 0)
            
            # 獲取對應頁面的原始內容
            original_content = original_contents.get(page_num, "")
            
            if original_content:
                # 計算精確的行號
                start_line, end_line = self._calculate_line_numbers(
                    original_content, 
                    text_chunk.page_content, 
                    start_index
                )
            else:
                # 後備方案
                lines = text_chunk.page_content.split('\n')
                start_line = 1
                end_line = len(lines)
            
            # 創建 source_location
            source_location = f"{filename}-P{page_num}-L{start_line}-L{end_line}"
            
            # 更新元數據
            text_chunk.metadata.update({
                "source_location": source_location,
                "chunk_index": i,
                "start_line": start_line,
                "end_line": end_line,
                "line_count": end_line - start_line + 1
            })

        # 步驟 4: 檢查並初始化向量存儲
        if self.vector_store is None:
            self.vector_store = self._load_or_create_vector_store()
            if self.vector_store is None:
                log.error("Failed to initialize vector store, cannot add documents.")
                return

        # 步驟 5: 將新的文件區塊加入到現有的向量儲存中
        log.info(f"Adding {len(texts)} new text chunks from '{filename}' to the vector store.")
        self.vector_store.add_documents(texts)

        # 步驟 6: 將更新後的索引儲存回本地磁碟
        self.vector_store.save_local(rag_config.vector_store_path)
        log.info(f"Vector store updated and saved to {rag_config.vector_store_path}")
        log.info("QA chain has been re-initialized with the updated vector store.")

    def get_page_content_by_filename(self, filename: str) -> list[str]:
        """
        根據檔名，從向量資料庫中讀取所有相關的 page_content。

        Args:
            filename (str): 要查詢的檔案名稱。

        Returns:
            list[str]: 一個包含所有符合條件的 page_content 的列表。
        """
        if not self.vector_store:
            log.warning("Vector store is not initialized. Cannot retrieve content.")
            return []

        if not hasattr(self.vector_store, 'docstore') or not hasattr(self.vector_store.docstore, '_dict'):
            log.error("The vector store's docstore is not accessible in the expected format.")
            return []

        all_docs = self.vector_store.docstore._dict.values()
        
        matching_content = [(doc.page_content, doc.id) for doc in all_docs if doc.metadata.get('source') == filename]
        
        if not matching_content:
            log.info(f"No content found for filename: {filename}")
        else:
            log.info(f"Found {len(matching_content)} content chunks for filename: {filename}")

        return matching_content
    
    def list_documents(self) -> list[str]:
        """
        列出知識庫資料夾中的所有檔案。

        Returns:
            list[str]: 一個包含所有檔案名稱的列表。
        """
        if not os.path.isdir(rag_config.docs_path):
            log.warning(f"知識庫目錄不存在: {rag_config.docs_path}")
            return []
        
        try:
            # 使用列表推導式來獲取所有檔案的名稱
            return [f for f in os.listdir(rag_config.docs_path) if os.path.isfile(os.path.join(rag_config.docs_path, f))]
        except Exception as e:
            log.error(f"列出文件時發生錯誤: {e}")
            return []

    def delete_document(self, filename: str) -> bool:
        """
        從知識庫資料夾和向量資料庫中，安全地刪除一個檔案及其所有相關資料。

        Args:
            filename (str): 要刪除的檔案名稱。

        Returns:
            bool: 如果成功刪除則返回 True，否則返回 False。
        """
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            log.warning(f"偵測到潛在的路徑遍歷攻擊，已拒絕刪除請求: {filename}")
            return False

        # 步驟 1: 從向量資料庫中刪除對應的向量
        if self.vector_store and hasattr(self.vector_store, 'docstore'):
            ids_to_delete = [
                doc_id for doc_id, doc in self.vector_store.docstore._dict.items()
                if doc.metadata.get('source') == safe_filename
            ]

            if ids_to_delete:
                try:
                    self.vector_store.delete(ids_to_delete)
                    log.info(f"已成功從向量資料庫中刪除 {len(ids_to_delete)} 個與 '{safe_filename}' 相關的區塊。")
                    self.vector_store.save_local(rag_config.vector_store_path)
                    log.info(f"刪除後已更新並儲存向量資料庫至 {rag_config.vector_store_path}")
                except Exception as e:
                    log.error(f"從向量資料庫刪除 '{safe_filename}' 的向量時發生錯誤: {e}")
                    return False
            else:
                log.info(f"在向量資料庫中找不到與 '{safe_filename}' 相關的向量。")
        else:
            log.warning("向量資料庫未初始化或格式不符，無法刪除向量。")

        # 步驟 2: 從檔案系統中刪除原始檔案
        file_path = os.path.join(rag_config.docs_path, safe_filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                log.info(f"已成功刪除原始檔案: {file_path}")
                return True
            else:
                log.warning(f"嘗試刪除但找不到原始檔案: {file_path}")
                # 即使找不到檔案，如果向量已刪除，也可能視為部分成功
                return True
        except Exception as e:
            log.error(f"刪除檔案 '{safe_filename}' 時發生錯誤: {e}")
            return False


    def _load_or_create_vector_store(self):
        """
        Loads the vector store from disk if it exists, 
        otherwise creates a new one.
        """
        if os.path.exists(rag_config.vector_store_path) and os.listdir(rag_config.vector_store_path):
            log.info(f"Loading existing vector store from {rag_config.vector_store_path}")
            return FAISS.load_local(
                rag_config.vector_store_path,
                self.embed_model,
                allow_dangerous_deserialization=True # FAISS需要此參數
            )
        else:
            log.info("No existing vector store found. Creating a new one.")
            # 建立一個空的向量儲存，之後可以動態加入文件
            # 我們需要一個初始的假文件來建立索引
            initial_doc = [Document(page_content="system initialization")]
            store = FAISS.from_documents(initial_doc, self.embed_model)
            store.save_local(rag_config.vector_store_path)
            return store

    def add_document(self, content: str, filename: str):
        """
        Adds a new document to the vector store without rebuilding.
        1. Saves the raw text file.
        2. Splits the text into chunks.
        3. Adds the document chunks to the existing FAISS index.
        4. Saves the updated index to disk.
        """
        # 1. 儲存原始的 .txt 記憶文件
        file_path = os.path.join(rag_config.docs_path, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            log.info(f"Saved new memory file: {filename}")
        except IOError as e:
            log.error(f"Failed to save memory file {filename}: {e}")
            return

        # 2. 建立 LangChain 的 Document 物件
        # 注意：我們直接從內容建立，而不是從檔案載入，這樣更直接
        docs = [Document(page_content=content, metadata={"source": filename})]
        texts = self.text_splitter.split_documents(docs)
        
        # Check if vector store is None and initialize it if necessary
        if self.vector_store is None:
            log.info("Vector store is None, attempting to initialize it...")
            self.vector_store = self._load_or_create_vector_store()
            if self.vector_store is None:
                log.error("Failed to initialize vector store, cannot add documents.")
                return
        
        # 3. 將新的文件區塊加入到現有的向量儲存中
        log.info(f"Adding {len(texts)} new text chunks to the vector store.")
        self.vector_store.add_documents(texts)
        
        # 4. 將更新後的索引儲存回本地磁碟
        self.vector_store.save_local(rag_config.vector_store_path)
        log.info(f"Vector store updated and saved to {rag_config.vector_store_path}")

    def rebuild_vector_store(self):
        """
        (核心功能) 清除舊的向量資料庫，並從 docs 目錄中的所有文件重新建立。
        """
        log.info("開始重建向量資料庫...")
        
        loader = DirectoryLoader(
            rag_config.docs_path, 
            glob="**/*.*",
            loader_cls=lambda path: UnstructuredFileLoader(path),
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()

        if not documents:
            log.warning(f"在 '{rag_config.docs_path}' 中找不到任何文件。將會清除舊的向量資料庫 (如果存在)。")
            if os.path.exists(rag_config.vector_store_path):
                shutil.rmtree(rag_config.vector_store_path)
                os.makedirs(rag_config.vector_store_path)
            self.vector_store = None
            self.qa_chain = None
            return

        docs = self.text_splitter.split_documents(documents)
        log.info(f"已將 {len(documents)} 個文件切割成 {len(docs)} 個區塊。")

        self.vector_store = FAISS.from_documents(docs, self.embed_model)
        self.vector_store.save_local(rag_config.vector_store_path)
        log.info(f"新的向量資料庫已成功建立並儲存至 {rag_config.vector_store_path}")

    def _calculate_line_numbers(self, original_content: str, chunk_content: str, start_index: int = 0):
        """
        計算文本塊在原始文檔中的精確行號範圍
        
        Args:
            original_content (str): 原始文檔內容
            chunk_content (str): 文本塊內容
            start_index (int): 文本塊在原始內容中的開始位置
            
        Returns:
            tuple: (start_line, end_line)
        """
        # 將原始內容按行分割
        lines = original_content.split('\n')
        
        # 計算到 start_index 位置為止有多少行
        char_count = 0
        start_line = 1
        
        for i, line in enumerate(lines):
            if char_count + len(line) + 1 > start_index:  # +1 for newline
                start_line = i + 1
                break
            char_count += len(line) + 1
        
        # 計算文本塊包含多少行
        chunk_lines = len(chunk_content.split('\n'))
        end_line = start_line + chunk_lines - 1
        
        return start_line, end_line
