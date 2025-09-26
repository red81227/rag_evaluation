# RAG 評估工具 (RAG Evaluation Tool)

這是一個用於評估現有 RAG (Retrieval-Augmented Generation) 系統效能的工具。它可以幫助使用者透過量化指標，了解其 RAG 系統在特定知識庫下的檢索品質。

## 專案功能

此專案在現有的 RAG 系統上外加一個評估服務，透過計算以下兩個關鍵指標來量化 RAG 系統的表現：

* **Hit Rate (命中率):** 評估系統對於一個問題，是否能成功召回至少一個相關的資訊片段（與黃金答案相似度高於閾值）。
* **Mean Reciprocal Rank (MRR, 平均倒數排名):** 衡量找到最相關資訊片段的效率。如果最相關的片段排在第一個，得分為 1；排在第二個，得分為 1/2，依此類推。此指標反映了系統能否將最相關的答案排在前面。

## 專案結構

```
.
├── rag_evaluation/                   # 評估工具的核心模組
│   ├── files/                        # 存放知識庫文件 (例如: PDF, TXT)
│   │   └── 1130318轉直供營運規章.pdf
│   ├── qas/                          # 存放根據知識庫產生的問答對 (golden samples)
│   │   └── qa_1130318轉直供營運規章.md
│   ├── prompts/                      # 用於生成問答對的提示範本
│   │   └── generate_aq_prompt.md
│   ├── rag_evaluate.py               # 評估器主要邏輯 (RAGEvaluator class)
│   └── utils.py                      # 工具函式 (例如: 讀取檔案)
│
└── your_project/                     # 一個模擬的 RAG 服務專案 (供參考)
    ├── configs/
    │   └── project_setting.py
    └── services/
        └── rag_service.py
```

* `rag_evaluation`: 評估工具的核心程式碼。
* `your_project`: 一個模擬的 RAG 服務，展示如何將評估工具整合到您自己的專案中。

## 使用步驟

### 1. 準備您的 RAG 服務

`your_project/services/rag_service.py` 是一個範例 RAG 服務，您可以參考其結構，或直接替換成您自己的 RAG 服務。重要的是，您的 `RAGService` 類別需要包含以下方法，以便評估器呼叫：

* `add_document_and_update_store(content_bytes: bytes, filename: str)`: 將文件加入知識庫並更新向量儲存。
* 一個可供存取的 `vector_store` 物件，且該物件需有 `similarity_search(query, k)` 方法。

### 2. 整合評估工具

將 `rag_evaluation` 資料夾複製到您的專案根目錄。

### 3. 執行評估

在您的專案中，您可以透過以下方式來執行評估：

```python
from rag_evaluation.rag_evaluate import RAGEvaluator
# 引用您專案中定義的 RAGService
from your_project.services.rag_service import RAGService

# 1. 實例化您的 RAG 服務
rag_service = RAGService()

# 2. 實例化評估器，並傳入您的 RAG 服務與評估資料路徑
# file_path 應指向 rag_evaluation 資料夾
evaluator = RAGEvaluator(rag_service=rag_service, file_path="./rag_evaluation")

# 3. 執行評估 (可傳入 k 值，代表 top-k 的檢索結果)
hit_rate, mrr = evaluator.evaluate(k=3)

# 4. 輸出評估結果
print(f"Hit Rate: {hit_rate:.2%}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
```

### 4. 準備您自己的評估資料
您可以參考 rag_evaluation/files 和 rag_evaluation/qas 中的範例，準備您自己的知識庫文件和對應的問答集，以更貼近您的實際應用場景。

* 知識庫 (/files): 放入您希望 RAG 系統學習的原始文件。

* 問答集 (/qas): 根據知識庫文件，建立一系列的問題與標準答案 (golden_source)。格式請參考範例 qa_*.md。
