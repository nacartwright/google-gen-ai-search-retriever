from __future__ import annotations, absolute_import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from typing import Any, Dict, List, Optional
from google.protobuf.json_format import MessageToDict
from google.cloud import discoveryengine_v1
from google.cloud.discoveryengine_v1.services.search_service import pagers
from langchain.schema import BaseRetriever, Document
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, Extra, Field, root_validator
from langchain.vectorstores import FAISS
from langchain.embeddings import VertexAIEmbeddings

# Create a text splitter that will split documents into chunks of 1000 characters,

text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len)

# Embeddings engine
embeddings = VertexAIEmbeddings()

class EnterpriseSearchRetriever(BaseRetriever, BaseModel):
    """Wrapper around Google Cloud Enterprise Search Service API."""

    _client: Any
    _serving_config: Any
    project_id: str # Project Number
    search_engine_id: str
    serving_config_id: str = "default_search"
    location_id: str = "global"
    filter: Optional[str] = None
    get_extractive_answers: bool = False
    max_documents: int = Field(default=5, ge=1, le=100)
    max_extractive_answer_count: int = Field(default=1, ge=1, le=5)
    max_extractive_segment_count: int = Field(default=1, ge=1, le=1)
    query_expansion_condition: int = Field(default=1, ge=0, le=2)
    credentials: Any = None
    "The default custom credentials (google.auth.credentials.Credentials) to use "
    "when making API calls. If not provided, credentials will be ascertained from "
    "the environment."

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["project_id"] = get_from_dict_or_env(values, "project_id", "PROJECT_ID")
        values["search_engine_id"] = get_from_dict_or_env(values, "search_engine_id", "SEARCH_ENGINE_ID")
        return values

    def __init__(self, **data):
        super().__init__(**data)
        self._client = discoveryengine_v1.SearchServiceClient()

    @classmethod
    def _get_web_document(self, urls: list) -> List[Document]:
        loader = WebBaseLoader(urls)
        documents = loader.load()
        return [Document(page_content=" ".join(doc.page_content.split()), metadata=doc.metadata) for doc in documents]
    
    @classmethod   
    def _chunk_documents(self, documents):
        chunks = text_splitter.split_documents(documents)
        return chunks
        
    @classmethod
    def _parse_search_response(self, urls: dict) -> List[str]:
        documents = urls["results"]
        og_urls = []

        for result in documents:
            metatags = result["document"]["derivedStructData"]["pagemap"]["metatags"]
            for tag in metatags:
                if "og:url" in tag:
                    og_urls.append(tag["og:url"])
        return og_urls
    
    def list_available_documents(self) -> List[Document]:
        self._client.lis

    def get_relevant_documents(self, query: str) -> List[Document]:
        try:        
            request = discoveryengine_v1.SearchRequest({
                "query": query,
                "serving_config": f"projects/{self.project_id}/locations/global/collections/default_collection/dataStores/{self.search_engine_id}/servingConfigs/{self.serving_config_id}",
                "page_size": self.max_documents})

            response = self._client.search(request=request)
            urls = self._parse_search_response(MessageToDict(response._pb))
            documents = self._get_web_document(urls)
            chunks = self._chunk_documents(documents)

            db = FAISS.from_documents(chunks, embeddings)
            retriever = db.as_retriever()
            return retriever.get_relevant_documents(query)

        except Exception as e:
            raise Exception(e)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async interface to GDELT not implemented")