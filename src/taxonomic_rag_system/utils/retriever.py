"""
Taxonomic RAG System retrievers and chain models.

This module provides utilities for building and managing a Taxonomic RAG system.
It includes classes for constructing RAG chains, retrieving documents, and
generating taxonomic classifications based on captions and contextual information.

Classes:
    RAGChainBuilder: Build and manage a Taxonomic RAG chain for
        taxonomic classification tasks.
    BaseRetriever: Base class for document retrieval using a Chroma collection.
    WikiStellaRAGModel: A specialized RAG model using Stella embeddings
        and Chroma for document retrieval.

Dependencies:
    - torch
    - pathlib
    - langchain
    - langchain_community
    - langchain_core
    - langchain_openai
    - taxonomic_rag_system.utils.base_models
    - taxonomic_rag_system.utils.helpers
"""

import os
import logging
from pathlib import Path
from typing import Any

import torch
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cohere_rerank import CohereRerank
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

# Local imports
from taxonomic_rag_system.utils.helpers import format_docs, unique_docs
from taxonomic_rag_system.utils.out_models import MultiQuery, TaxBiodiversity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_api_keys() -> None:
    """Load API keys from files."""
    # Original code, change back later
    # Set API key env variables w/ `.openai.key` and `.openrouter.key` files in home dir
    # with open(Path.home() / ".openai.key", "r") as f:
    #     os.environ["OPENAI_API_KEY"] = f.read().strip()
    # with open(Path.home() / ".openrouter.key", "r") as f:
    #     os.environ["OPENROUTER_API_KEY"] = f.read().strip()
    # with open(Path.home() / ".cohere.key", "r") as f:
    #     os.environ["COHERE_API_KEY"] = f.read().strip()

    # OpenAI API key is required
    try:
        with open(Path.home() / ".openai.key", "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find OpenAI API key at ~/.openai.key. "
            "This key is required for the model to function."
        )

    # OpenRouter API key is optional
    try:
        with open(Path.home() / ".openrouter.key", "r") as f:
            os.environ["OPENROUTER_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        # OpenRouter key is optional, only log a warning
        print("Warning: Could not find OpenRouter API key at ~/.openrouter.key. "
              "This is fine if you're not using OpenRouter models.")

    # Cohere API key is optional
    try:
        with open(Path.home() / ".cohere.key", "r") as f:
            os.environ["COHERE_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        # Cohere key is optional, only log a warning
        print("Warning: Could not find Cohere API key at ~/.cohere.key. "
              "This is fine if you're not using reranking functionality.")


class RAGChainBuilder:
    """
    Build and manage a Taxonomic RAG chain.

    The RAGChainBuilder class uses a retriever to manage a RAG chain with doc retrieval
    and taxonomic classification generation based on a caption and additional context.
    """

    def __init__(self, retriever: Any):
        """
        Initialize RAGChainBuilder with a retriever.

        :param retriever: The document retriever to use.
        """
        self.llm = ChatOpenAI(model="gpt-4o")
        self.output_parser = PydanticOutputParser(pydantic_object=TaxBiodiversity)
        self.system_template = (
            """
        You are an expert AI taxonomist. Your task is to use the organisms discussed in the caption of a new organism to generate a taxonomic classification for the new organism.

        You will also be provided with some context that could or could not match the caption, if there is information in the context that matches the caption, you can use that info to inform your decision about the taxonomic classification, otherwise, if information does not match the details provided in the caption, disregard it.

        You will be provided with context and a caption, provide in your response:
        1. A Taxonomic classification
        2. A description pairing physical traits common to both the new organism described in the caption and other organisms described in the context that indicate and support the choice made in the taxonomic classification.
        3. A description of physical traits particular to this new organism described in the caption. These traits may set it apart from other organisms, may suggest it has unique features, and/or contain traits that may be candidates to investigate for a more specific taxonomic classification.
        4. Commentary on your choice, including discussion of confidence, what new information about the new organism would help support the taxonomic classification and what new information would dispute the taxonomic classification.
        5. A few paragraphs describing the features present (from the caption) and how they could relate to the biodiversity knowledge that is relevant to the taxa chosen.

        Do not include a taxonomic classification for a certain rank unless you are confident from the caption (and/or context) about the classification.

        """
            + "..." * 256
            + """
        <context>
        {context}
        </context>

        <caption>
        {caption}
        </caption>

        {format_instructions}
        """
        )
        self.prompt = self._prompt_construction()
        self.rag_chain = self._build_chain(retriever)

    def _prompt_construction(self) -> PromptTemplate:
        """
        Construct the prompt template required for taxonomic classification tasks.

        :return: A structured prompt template.
        """
        return PromptTemplate(
            template=self.system_template,
            input_variables=["context", "caption"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            },
        )

    def _build_chain(
        self, retriever: Runnable[str, list[Document]] | VectorStoreRetriever
    ) -> RunnableSerializable[Any, TaxBiodiversity]:
        """
        Construct a RAG chain using the provided retriever and a prompt.

        :return: The constructed RAG chain.
        """
        return (
            {"context": retriever | format_docs, "caption": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.output_parser
        )

    def invoke(self, inp: dict[str, Any]) -> TaxBiodiversity:
        """
        Invoke the RAG chain synchronously with the given input.

        :param inp: The input data.
        :return: The output of the RAG chain.
        """
        return self.rag_chain.invoke(input=inp)

    async def ainvoke(self, inp: dict[str, Any]) -> TaxBiodiversity:
        """
        Invoke the RAG chain asynchronously with the given input.

        :param inp: The input data.
        :return: The output of the RAG chain.
        """
        return await self.rag_chain.ainvoke(input=inp)


class BaseRetriever:
    """Base class for document retriever against a Chroma collection."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        search_type: str,
        k: int,
    ):
        """
        Initialize a BaseRetriever for document retrieval with additional params.

        :param collection_name: The name of the collection to search.
        :param embedding_model: The embedding model to use.
        :param search_type: The type of search to perform.
        :param k: The number of results to return.
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.search_type = search_type
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WikiStellaRAGModel(BaseRetriever):
    """Set up a RAG Model using the Chroma collection and Stella embeddings."""

    def __init__(
        self,
        vstore_path: str,
        collection_name: str = "Wiki_contexted",
        embedding_model: str = "dunzhang/stella_en_1.5B_v5",
        search_type: str = "similarity",
        k: int = 30,
        rerank: bool = False,
        multiquery: bool = False,
    ):
        """
        Initialize WikiStellaRAGModel with config options for multiquery and reranker.

        :param vstore_path: Path to the chroma parent dir for the vector store.
        :param collection_name: The Chroma collection name for vectorstore.
            (default is 'Wiki_contexted').
        :param embedding_model: The name of the embedding model to use.
            (default is 'dunzhang/stella_en_1.5B_v5').
        :param search_type: The type of search to perform
            (default is 'similarity').
        :param k: Number of top results to retrieve
            (default is 30).
        :param rerank: Boolean flag to indicate if reranking
            should be applied (default is False).
        :param multiquery: Boolean flag to indicate if multi-query
            should be used (default is False).
        """
        super().__init__(
            collection_name=collection_name,
            embedding_model=embedding_model,
            search_type=search_type,
            k=k,
        )
        load_api_keys()
        self.vectorstore = self._set_up_retriever(vstore_path)
        self.base_retriever = self.vectorstore.as_retriever(
            search_type=self.search_type, search_kwargs={"k": self.k}
        )
        self.retriever: RunnableSerializable[Any, list[Document]] = self.base_retriever
        if multiquery:
            self._add_multiquery()
        if rerank:
            self._add_reranker()
        self.model = RAGChainBuilder(self.retriever)

    def _set_up_retriever(self, vstore_path: str) -> Chroma:
        """
        Set up the vector store retriever.

        Configure the retriever with the specified path to chroma parent dir.
        The logic is:
        1. Try to load the existing collection
        2. If the collection does not exist, create a new one with embeddings

        :return: A configured Chroma vector store.
        """

        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=vstore_path
            )
            logger.info(f"Successfully loaded vector store from {vstore_path}")
            return vectorstore
        except ValueError:
            # Collection does not exist – need to create new one (requires embeddings)
            logger.info("Collection not found; creating new vector store with embeddings …")
            encode_kwargs = {
                "normalize_embeddings": True,  # Use faster dot-product instead cosine sim
                "batch_size": 128,
            }
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs=encode_kwargs,
                show_progress=False,
            )
            return Chroma(
                embedding_function=embeddings,
                persist_directory=vstore_path,
                collection_name=self.collection_name,
            )

    def _add_reranker(self, top_n: int = 10) -> None:
        """
        Add reranking capability to the retriever using a document compressor.

        :param top_n: Number of top documents to retain after reranking (default is 10).
        """
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=top_n)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.base_retriever
        )

    def _add_multiquery(self) -> None:
        """
        Add multi-query capability to the retriever.

        This generates semantically varied queries to improve retrieval
        performance.
        """
        system_template = """
        You are an AI language model assistant. Your task is generate 3 questions to be used in semantic similarity search queries that will
        help with taxonomic classification of the primary organism in a user-provided image caption.

        By generating multiple perspectives, your goal is to help the user with semantic search
        to a database in fetching context for taxonomically identifying an organism and describing
        it in ways relevant for the study of biodiversity.

        Generate 1 question aimed at each of the following:
        1. External morphology or feature descriptions
        2. Context, abitotic and biotic relationships, surroundings or habitat
        3. Action, function or ecological role/niche

        {format_instructions}

        Original caption: {caption}
        """
        # Create Pydantic parser and prompt for parsing into multiple queries
        parser = PydanticOutputParser(pydantic_object=MultiQuery)
        prompt_perspectives = PromptTemplate(
            template=system_template,
            input_variables=["caption"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        generate_queries = (
            prompt_perspectives
            | ChatOpenAI(model="gpt-4o-mini")
            | parser
            | (lambda x: x.queries)
        )
        # Redefine retriever to use union of output from multiple retrievals
        self.retriever = generate_queries | self.retriever.map() | unique_docs

    def retrieve(self, caption: str) -> list[Document]:
        """
        Retrieve documents using configured retriever with a caption.

        :param caption: The caption to use for retrieval.
        :return: Retrieved documents.
        """
        return self.retriever.invoke(input=caption)

    async def aretrieve(self, caption: str) -> list[Document]:
        """
        Asynchronously retrieves documents using configured retriever with a caption.

        :param caption: The caption to use for retrieval.
        :return: Retrieved documents.
        """
        return await self.retriever.ainvoke(input=caption)

    def invoke(self, caption: str) -> TaxBiodiversity:
        """
        Invoke the RAG model synchronously using the given caption.

        :param caption: The caption to classify and describe taxonomically.
        :return: The output of the RAG model with taxonomic information.
        """
        # Retrieve documents using the retriever
        docs = self.retrieve(caption=caption)
        # Format the retrieved documents
        formatted_docs = format_docs(docs)
        # Create the input for the RAG model
        inp = {"context": formatted_docs, "caption": caption}
        # Invoke RAG model and return results
        return self.model.invoke(inp=inp)

    async def ainvoke(self, caption: str) -> TaxBiodiversity:
        """
        Invoke the RAG model asynchronously using the given caption.

        :param caption: The caption to classify and describe taxonomically.
        :return: The output of the RAG model with taxonomic information.
        """
        # Retrieve documents using the retriever
        docs = await self.aretrieve(caption=caption)
        # Format the retrieved documents
        formatted_docs = format_docs(docs)
        # Create the input for the RAG model
        inp = {"context": formatted_docs, "caption": caption}
        # Invoke RAG model and return results
        return await self.model.ainvoke(inp=inp)
