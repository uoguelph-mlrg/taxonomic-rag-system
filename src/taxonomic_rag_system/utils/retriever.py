"""
Taxonomic RAG System retrievers and chain models.

This module provides utilities for building and managing a Taxonomic RAG system.
It includes classes for constructing RAG chains, retrieving documents, and
generating taxonomic classifications based on captions and contextual information.

Classes:
    SafeHuggingFaceEmbeddings: A custom safety wrapper for HuggingFaceEmbeddings
    RAGChainBuilder: Build and manage a Taxonomic RAG chain for
        taxonomic classification tasks.
    BaseRetriever: Base class for document retrieval using a Chroma collection.
    WikiStellaRAGModel: A specialized RAG model using Stella embeddings
        and Chroma for document retrieval.

Dependencies:
    - torch
    - langchain
    - langchain_community
    - langchain_core
    - langchain_openai
    - taxonomic_rag_system.utils.out_models
    - taxonomic_rag_system.utils.helpers
"""

import logging
import hashlib
import traceback
from typing import Any, Union, overload

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
import json  # For saving prompt/response pairs
from pathlib import Path  # Handle log file paths

# Local imports
from taxonomic_rag_system.utils.helpers import format_docs, load_api_keys, unique_docs
from taxonomic_rag_system.utils.out_models import MultiQuery, TaxBiodiversity


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed seed for LLM requests (reproducibility/audit)
DEFAULT_LLM_SEED: int = 12345


class SafeHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    A safe wrapper around HuggingFaceEmbeddings that ensures all inputs are strings.

    This prevents 'dict' object has no attribute 'replace' errors by converting
    any non-string inputs to strings before passing them to the underlying embeddings.
    """

    @overload
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @overload
    def embed_documents(self, texts: list[dict[str, str]]) -> list[list[float]]: ...

    @overload
    def embed_documents(self, texts: list[Any]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[Any]) -> list[list[float]]:
        """Safely embed documents by ensuring all inputs are strings."""
        safe_texts: list[str] = []
        for text in texts:
            if isinstance(text, str):
                safe_texts.append(text)
            elif isinstance(text, dict):
                safe_texts.append(str(text))
            else:
                safe_texts.append(str(text))
        return super().embed_documents(safe_texts)

    @overload
    def embed_query(self, text: str) -> list[float]: ...

    @overload
    def embed_query(self, text: dict[str, str]) -> list[float]: ...

    def embed_query(self, text: Union[str, dict[str, str]]) -> list[float]:
        """Safely embed a query by ensuring the input is a string."""
        if isinstance(text, str):
            safe_text = text
        elif isinstance(text, dict):
            safe_text = str(text)
        else:
            safe_text = str(text)
        return super().embed_query(safe_text)


class RAGChainBuilder:
    """
    Build and manage a Taxonomic RAG chain.

    The RAGChainBuilder class uses a retriever to manage a RAG chain with doc retrieval
    and taxonomic classification generation based on a caption and additional context.
    """

    def __init__(self, retriever: Any, log_path: str | None = None, logprobs_path: str | None = None) -> None:
        """
        Initialize RAGChainBuilder with a retriever.

        :param retriever: The document retriever to use.
        :param log_path: Optional path to save prompt/response pairs in JSONL format.
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
        # Optional path to save prompt/response pairs in JSONL format; disabled by default
        self._log_path: str | None = str(log_path) if log_path else None
        # Optional path to save token-level logprobs JSONL aligned by rsid
        self._logprobs_path: str | None = str(logprobs_path) if logprobs_path else None
        # Store retriever for possible prompt reconstruction/logging
        self._retriever = retriever
    
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
        # Prompt/Response logging
        # Use only required keys for formatting to avoid extra keys issues
        prompt_inputs = {k: inp[k] for k in ("context", "caption") if k in inp}
        prompt_str = self.prompt.format(**prompt_inputs)
        # Build a generation runnable that requests logprobs and enforces JSON
        gen = self.prompt | self.llm.bind(
            logprobs=True,
            response_format={"type": "json_object"},
            temperature=1,
            top_p=1,
            seed=DEFAULT_LLM_SEED,
        )
        # Invoke model to get AIMessage with response metadata (incl. logprobs)
        ai_msg = gen.invoke(input=inp)
        # Parse JSON to pydantic object using the same parser as before
        result = self.output_parser.invoke(ai_msg.content)
        # Log prompt/response pair with optional RSID
        self._log_pair(prompt_str, result, rsid=inp.get("RSID"))
        # Also write logprobs JSONL if configured
        self._log_logprobs(prompt_str, ai_msg, rsid=inp.get("RSID"))
        return result

    async def ainvoke(self, inp: dict[str, Any]) -> TaxBiodiversity:
        """
        Invoke the RAG chain asynchronously with the given input.

        :param inp: The input data.
        :return: The output of the RAG chain.
        """
        try:
            # Prompt/Response logging parity with sync invoke
            prompt_inputs = {k: inp[k] for k in ("context", "caption") if k in inp}
            prompt_str = self.prompt.format(**prompt_inputs)
            # Build a generation runnable that requests logprobs and enforces JSON
            gen = self.prompt | self.llm.bind(
                logprobs=True,
                response_format={"type": "json_object"},
                temperature=1,
                top_p=1,
                seed=DEFAULT_LLM_SEED,
            )
            # Invoke model asynchronously to get AIMessage
            ai_msg = await gen.ainvoke(input=inp)
            # Parse JSON to pydantic object
            result = self.output_parser.invoke(ai_msg.content)
            # Log prompt/response pair
            self._log_pair(prompt_str, result, rsid=inp.get("RSID"))
            # Also write logprobs JSONL if configured
            self._log_logprobs(prompt_str, ai_msg, rsid=inp.get("RSID"))
            return result
        except Exception as e:
            logger.error("Error in RAGChainBuilder.ainvoke:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            # Return a default result instead of letting the error propagate
            return TaxBiodiversity(
                classification={
                    "Kingdom": "Animalia",
                    "Phylum": "N/A",
                    "Class": "N/A",
                    "Order": "N/A",
                    "Family": "N/A",
                    "Genus": "N/A",
                    "Species": "N/A",
                },
                ancestral="Error occurred during processing",
                specific="Error occurred during processing",
                commentary="Error occurred during processing",
                bio_knowledge="Error occurred during processing",
            )
    
    def _log_pair(self, prompt: str, response: Any, rsid: str | None = None) -> None:
        """Append a prompt/response pair to the configured JSONL log file (if any).

        Each line in the file is a JSON object with keys ``prompt`` and ``response``.

        The response is serialised as a dict via ``model_dump`` when the object comes
        from a Pydantic model; otherwise ``str(response)`` is used.
        """
        if not getattr(self, "_log_path", None):
            return  # Logging disabled

        try:
            # Ensure parent directory exists
            Path(self._log_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

            if hasattr(response, "model_dump"):
                resp_obj = response.model_dump()
            else:
                resp_obj = str(response)

            with open(self._log_path, "a", encoding="utf-8") as fp:
                # Keep lowercase 'rsid' as the first key, followed by 'prompt' and 'response'
                json_line = json.dumps({"rsid": rsid, "prompt": prompt, "response": resp_obj}, ensure_ascii=False)
                fp.write(json_line + "\n")
        except Exception as log_err:
            # Do not crash the main pipeline for logging errors; just warn.
            logger.warning(f"Failed to log prompt/response pair: {log_err}")

    def _log_logprobs(self, prompt: str, ai_message: Any, rsid: str | None = None) -> None:
        """Append a token-level logprobs record to the configured JSONL file (if any).

        The record includes:
          - rsid, model, created, finish_reason
          - response_text and its sha256 hash
          - tokens with per-token logprob and char/byte offsets
          - section mappings: top-level fields and classification ranks
        """
        if not getattr(self, "_logprobs_path", None):
            return  # Logging disabled

        try:
            # Ensure parent directory exists
            Path(self._logprobs_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

            # Extract metadata and tokens from AIMessage
            full_text = getattr(ai_message, "content", "") or ""
            meta = getattr(ai_message, "response_metadata", {}) or {}
            model_name = meta.get("model_name") or meta.get("model") or ""
            created = meta.get("created") or None
            finish_reason = meta.get("finish_reason") or (meta.get("choices", [{}])[0].get("finish_reason") if isinstance(meta.get("choices"), list) and meta.get("choices") else None)
            system_fingerprint = meta.get("system_fingerprint") or None

            # Try to extract tokens from various possible shapes
            tokens_src = []
            try:
                lp = meta.get("logprobs")
                if isinstance(lp, dict) and isinstance(lp.get("content"), list):
                    tokens_src = lp["content"]
                elif isinstance(meta.get("choices"), list):
                    ch0 = meta["choices"][0] if meta["choices"] else {}
                    lp2 = ch0.get("logprobs") if isinstance(ch0, dict) else None
                    if isinstance(lp2, dict) and isinstance(lp2.get("content"), list):
                        tokens_src = lp2["content"]
            except Exception:
                tokens_src = []

            # Build per-token records and compute offsets
            token_records: list[dict[str, object]] = []
            char_cursor = 0
            byte_cursor = 0
            rebuilt = []
            for i, tk in enumerate(tokens_src):
                tok = tk.get("token") if isinstance(tk, dict) else None
                lpv = tk.get("logprob") if isinstance(tk, dict) else None
                if tok is None:
                    continue
                s_char = char_cursor
                s_byte = byte_cursor
                rebuilt.append(tok)
                char_cursor += len(tok)
                byte_cursor += len(tok.encode("utf-8"))
                token_records.append({
                    "idx": i,
                    "t": tok,
                    "lp": lpv,
                    "char_s": s_char,
                    "char_e": char_cursor,
                    "byte_s": s_byte,
                    "byte_e": byte_cursor,
                })

            # Validate reconstructed text vs full_text (best effort)
            rebuilt_text = "".join(rebuilt)
            warnings: list[str] = []
            if full_text and rebuilt_text and rebuilt_text != full_text:
                warnings.append("rebuilt_text_mismatch")

            # Section mapping helpers
            def _find_json_string_value_span(raw: str, key: str, start_at: int = 0, end_at: int | None = None):
                end_lim = len(raw) if end_at is None else end_at
                kq = f'"{key}"'
                i = raw.find(kq, start_at, end_lim)
                if i == -1:
                    return None
                j = raw.find(":", i + len(kq), end_lim)
                if j == -1:
                    return None
                j += 1
                while j < end_lim and raw[j] in " \t\r\n":
                    j += 1
                if j >= end_lim or raw[j] != '"':
                    return None
                val_start = j + 1
                p = val_start
                while p < end_lim:
                    c = raw[p]
                    if c == "\\":
                        p += 2
                        continue
                    if c == '"':
                        return (val_start, p)
                    p += 1
                return None

            def _find_json_object_span(raw: str, key: str, start_at: int = 0):
                kq = f'"{key}"'
                i = raw.find(kq, start_at)
                if i == -1:
                    return None
                j = raw.find(":", i + len(kq))
                if j == -1:
                    return None
                j += 1
                while j < len(raw) and raw[j] in " \t\r\n":
                    j += 1
                if j >= len(raw) or raw[j] != '{':
                    return None
                # Brace matching with string awareness
                depth = 0
                in_str = False
                p = j
                while p < len(raw):
                    ch = raw[p]
                    if in_str:
                        if ch == "\\":
                            p += 2
                            continue
                        if ch == '"':
                            in_str = False
                        p += 1
                        continue
                    if ch == '"':
                        in_str = True
                        p += 1
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return (j, p + 1)
                    p += 1
                return None

            def _token_range_for_char_range(recs: list[dict[str, object]], cr: tuple[int, int] | None):
                if not cr:
                    return None
                s_char, e_char = cr
                idxs: list[int] = []
                for r in recs:
                    rs = int(r["char_s"])  # type: ignore[arg-type]
                    re = int(r["char_e"])  # type: ignore[arg-type]
                    if re > s_char and rs < e_char:
                        idxs.append(int(r["idx"]))  # type: ignore[arg-type]
                if not idxs:
                    return None
                return (min(idxs), max(idxs) + 1)

            sections: dict[str, object] = {}
            # Top-level string fields
            for key in ("ancestral", "specific", "commentary", "bio_knowledge"):
                cr = _find_json_string_value_span(full_text, key)
                tr = _token_range_for_char_range(token_records, cr) if token_records else None
                if cr is not None:
                    sections[key] = {"char_range": list(cr), "token_range": list(tr) if tr else None}

            # classification object and its ranks
            class_obj_span = _find_json_object_span(full_text, "classification")
            class_map: dict[str, object] = {}
            if class_obj_span is not None:
                c_s, c_e = class_obj_span
                for rk in ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"):
                    scr = _find_json_string_value_span(full_text, rk, start_at=c_s, end_at=c_e)
                    tr = _token_range_for_char_range(token_records, scr) if token_records else None
                    if scr is not None:
                        class_map[rk] = {"char_range": list(scr), "token_range": list(tr) if tr else None}
                sections["classification"] = class_map

            # Build record
            resp_hash = "sha256:" + hashlib.sha256(full_text.encode("utf-8")).hexdigest()
            record = {
                "schema_version": "logprob_v1",
                "rsid": rsid,
                "model": model_name,
                "created": created,
                "finish_reason": finish_reason,
                "system_fingerprint": system_fingerprint,
                "response_text": full_text,
                "response_text_hash": resp_hash,
                "gen_params": {"logprobs": True, "temperature": 1, "top_p": 1, "seed": DEFAULT_LLM_SEED},
                "tokens": token_records,
                "sections": sections,
                "warnings": warnings,
            }

            with open(self._logprobs_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_err:
            logger.warning(f"Failed to log logprobs: {log_err}")

class BaseRetriever:
    """Base class for document retriever against a Chroma collection."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        search_type: str,
        k: int,
    ) -> None:
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
        log_path: str | None = None,
        logprobs_path: str | None = None,
    ) -> None:
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
        self.model = RAGChainBuilder(self.retriever, log_path=log_path, logprobs_path=logprobs_path)

    def _set_up_retriever(self, vstore_path: str) -> Chroma:
        """
        Set up the vector store retriever.

        Configure the retriever with the specified path to chroma parent dir.
        The logic is:
        1. Try to load the existing collection
        2. If the collection does not exist, create a new one with embeddings

        :return: A configured Chroma vector store.
        """
        # Always create embeddings function for consistency
        encode_kwargs = {
            "normalize_embeddings": True,  # Use faster dot-product instead cosine sim
            "batch_size": 128,
        }
        embeddings = SafeHuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs=encode_kwargs,
            show_progress=False,
        )

        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=vstore_path,
                embedding_function=embeddings,
            )
            logger.info(f"Successfully loaded vector store from {vstore_path}")
            return vectorstore
        except ValueError:
            # Collection does not exist – need to create new one (requires embeddings)
            logger.info(
                "Collection not found; creating new vector store with embeddings …"
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

    def invoke(self, caption: str, RSID: str | None = None) -> TaxBiodiversity:
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
        if RSID is not None:
            inp["RSID"] = RSID
        # Invoke RAG model and return results
        return self.model.invoke(inp=inp)

    async def ainvoke(self, caption: str, RSID: str | None = None) -> TaxBiodiversity:
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
        if RSID is not None:
            inp["RSID"] = RSID
        # Invoke RAG model and return results
        return await self.model.ainvoke(inp=inp)
