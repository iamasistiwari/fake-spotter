# rag_model.py
import json
import logging
from typing import List, Dict, Union

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from web_scraper import GoogleSearchScraper, ContentScraper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_model")

# Output schema for fact-checking results
class FactCheckResult(BaseModel):
    claim: str = Field(description="The claim being verified")
    is_valid: bool = Field(description="Whether the claim is valid/true", default=None)
    confidence: str = Field(description="Confidence level (high, medium, low)")
    explanation: str = Field(description="Detailed explanation of the verdict")
    sources: List[Dict[str, str]] = Field(description="Sources used for verification")

class RAGFactChecker:
    def __init__(self, api_key=None):
        self.api_key = api_key
        try:
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=api_key
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI components: {e}")
            logger.warning("Continuing without LLM capabilities. This will impact results.")
            self.embeddings = None
            self.llm = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.google_scraper = GoogleSearchScraper()
        self.content_scraper = ContentScraper()
        self.vectorstore = None
        self.used_sources = []  # Track sources used for verification
    
    def _prepare_documents(self, content_list: List[Dict]) -> List[Document]:
        """Convert scraped content to Document objects for RAG"""
        documents = []
        
        for item in content_list:
            text = item["text"]
            metadata = {
                "source": item["source"],
                "url": item["url"],
                "title": item.get("title", ""),
                "publication_date": item.get("publication_date")
            }
            
            # Split text into chunks for better retrieval
            text_chunks = self.text_splitter.split_text(text)
            
            # Create Document objects
            for i, chunk in enumerate(text_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "chunk_id": i
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _create_vector_db(self, documents: List[Document]):
        """Create vector store from documents"""
        if self.embeddings is None:
            logger.error("Cannot create vector database without embeddings model")
            return False
            
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return True
    
    def create_rag_chain(self):
        """Create a RAG chain for fact checking"""
        if not self.vectorstore or not self.llm:
            logger.error("Cannot create RAG chain without vector store and LLM")
            return None
            
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create custom retriever that tracks used sources
        class SourceTrackingRetriever(BaseRetriever):
            def __init__(self, base_retriever, source_tracker):
                self.base_retriever = base_retriever
                self.source_tracker = source_tracker
                
            def get_relevant_documents(self, query):
                docs = self.base_retriever.get_relevant_documents(query)
                
                # Track sources used
                for doc in docs:
                    source_info = {
                        "title": doc.metadata.get("title", "Unknown"),
                        "url": doc.metadata.get("url", ""),
                        "source": doc.metadata.get("source", "Unknown"),
                        "publication_date": doc.metadata.get("publication_date")
                    }
                    
                    # Avoid duplicates
                    if source_info not in self.source_tracker:
                        self.source_tracker.append(source_info)
                
                return docs
                
        # Create tracking retriever
        tracking_retriever = SourceTrackingRetriever(retriever, self.used_sources)
        
        # Output parser for structured results
        output_parser = PydanticOutputParser(pydantic_object=FactCheckResult)
        
        # Create a prompt template for fact-checking
        template = """
        You are a fact-checking assistant tasked with verifying whether information is real or fake.
        
        User claim: {input}
        
        Based on the information from trusted sources below, determine if the user's claim 
        is valid/true or invalid/false. Be objective and evidence-based in your assessment.
        
        Sources:
        {context}
        
        Respond with a JSON object matching this format:
        ```json
        {
            "claim": "The claim being verified",
            "is_valid": true or false,
            "confidence": "high/medium/low",
            "explanation": "Your detailed explanation with specific evidence",
            "sources": [
                {
                    "title": "Source Title",
                    "url": "Source URL",
                    "source": "Domain name", 
                    "publication_date": "Date if available"
                }
            ]
        }
        ```
        
        Base your is_valid determination strictly on whether the claim is factually accurate according to the sources.
        Include all relevant sources that support your conclusion in the sources list.
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "input"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        
        # Create the document chain
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, prompt
        )
        
        # Create and return the final RAG chain
        return create_retrieval_chain(tracking_retriever, combine_docs_chain)
    
    def fact_check(self, query: str, search_term: str = None) -> Dict:
        """
        Fact check a user query or link.
        
        Args:
            query: The user query or claim to verify
            search_term: Optional search term to use instead of the query
        
        Returns:
            Dict containing the verification result with sources
        """
        # Reset used sources
        self.used_sources = []
        
        # Check if the query is a URL
        if query.startswith(("http://", "https://")):
            logger.info(f"Processing URL: {query}")
            content = self.content_scraper.scrape_url(query)
            if not content:
                return {
                    "claim": query,
                    "is_valid": False,
                    "confidence": "low",
                    "explanation": "Could not extract content from the provided URL.",
                    "sources": []
                }
            search_term = search_term or content[0].get("title", "")
        else:
            search_term = search_term or query
        
        # Get search results
        search_urls = self.google_scraper.search(search_term)
        
        if not search_urls:
            return {
                "claim": query,
                "is_valid": False,
                "confidence": "low",
                "explanation": "Could not find relevant information from trusted sources.",
                "sources": []
            }
        
        # Scrape content from search results
        content_list = self.content_scraper.scrape_multiple_urls(search_urls)
        
        if not content_list:
            return {
                "claim": query,
                "is_valid": False,
                "confidence": "low",
                "explanation": "Could not extract content from the search results.",
                "sources": []
            }
        
        # Prepare documents and create vector store
        documents = self._prepare_documents(content_list)
        if not self._create_vector_db(documents):
            return {
                "claim": query,
                "is_valid": None,
                "confidence": "low",
                "explanation": "Technical error: Could not create vector database for RAG processing.",
                "sources": []
            }
        
        # Create and run the RAG chain
        rag_chain = self.create_rag_chain()
        if not rag_chain:
            # Fall back to simple source listing if RAG chain creation fails
            sources = []
            for doc in documents[:5]:  # Just use the first few documents
                source_info = {
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("url", ""),
                    "source": doc.metadata.get("source", "Unknown"),
                    "publication_date": doc.metadata.get("publication_date")
                }
                if source_info not in sources:
                    sources.append(source_info)
            
            return {
                "claim": query,
                "is_valid": None,
                "confidence": "low",
                "explanation": "Could not analyze the content with RAG due to technical limitations. Found some potentially relevant sources.",
                "sources": sources
            }
        
        try:
            # Run the RAG chain
            result = rag_chain.invoke({"input": query})
            
            # Parse the response
            output = json.loads(result["answer"])
            
            # Ensure source information is included
            if not output.get("sources") and self.used_sources:
                output["sources"] = self.used_sources
                
            return output
        except Exception as e:
            logger.error(f"Error during RAG processing: {e}")
            # If processing fails, create a structured response
            return {
                "claim": query,
                "is_valid": None,  # Cannot determine
                "confidence": "low",
                "explanation": f"Error analyzing the results: {str(e)}",
                "sources": self.used_sources if self.used_sources else [{"title": "Source information unavailable", "url": "", "source": ""}]
            }