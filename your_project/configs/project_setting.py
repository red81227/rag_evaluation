"""This file is for application config"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict



class RAGConfig(BaseSettings):
    """Configuration for the RAG service."""
    vector_store_path: str = "../data/vector_store"
    docs_path: str = "../data/docs"
    model_name: str = Field(default="openai/text-embedding-3-small", validation_alias="RAG_MODEL_NAME")
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="allow"
    )

# 初始化設定
rag_config = RAGConfig()
