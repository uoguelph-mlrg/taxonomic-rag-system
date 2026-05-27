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

import hashlib
import json  # For saving prompt/response pairs
import logging
import traceback
from pathlib import Path  # Handle log file paths
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

# Local imports
from taxonomic_rag_system.utils.helpers import format_docs, load_api_keys, unique_docs
from taxonomic_rag_system.utils.logprobs_logging import (
    append_logprobs_from_langchain_message,
)
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

    def __init__(
        self,
        retriever: Any,
        log_path: str | None = None,
        logprobs_path: str | None = None,
        multisample_prompt_path: str | None = None,
        multisample_base_path: str | None = None,
        multisample_samples_path: str | None = None,
        multisample_logprobs_path: str | None = None,
    ) -> None:
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

        Provide your best-guess at every rank using scientific names and maintain lineage consistency (ancestors must be compatible with descendants). Prefer filling each rank over using 'N/A'; only output 'N/A' when there is truly no plausible guess from the caption and/or context. If uncertain, still provide the most probable taxon and explain uncertainty in commentary.
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
        # Optional paths for multi-sample prompt / sample / logprob logs.
        self._multisample_prompt_path: str | None = (
            str(multisample_prompt_path) if multisample_prompt_path else None
        )
        self._multisample_base_path: str | None = (
            str(multisample_base_path) if multisample_base_path else None
        )
        self._multisample_samples_path: str | None = (
            str(multisample_samples_path) if multisample_samples_path else None
        )
        self._multisample_logprobs_path: str | None = (
            str(multisample_logprobs_path) if multisample_logprobs_path else None
        )
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

    def _format_prompt(self, inp: dict[str, Any]) -> str:
        """Format the final prompt string for logging and hashing."""
        prompt_inputs = {k: inp[k] for k in ("context", "caption") if k in inp}
        return self.prompt.format(**prompt_inputs)

    def _build_generation_runnable(
        self,
        *,
        temperature: float = 1,
        top_p: float = 1,
        seed: int | None = DEFAULT_LLM_SEED,
        top_logprobs: int = 20,
        use_logprobs: bool = True,
    ) -> RunnableSerializable[Any, Any]:
        """Build a generation runnable with configurable sampling parameters."""
        bind_kwargs: dict[str, Any] = {
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "top_p": top_p,
        }
        if use_logprobs:
            bind_kwargs["logprobs"] = True
            bind_kwargs["top_logprobs"] = top_logprobs
        if seed is not None:
            bind_kwargs["seed"] = seed
        return self.prompt | self.llm.bind(**bind_kwargs)

    def _default_result(self) -> TaxBiodiversity:
        """Return a fallback result when generation or parsing fails."""
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

    def invoke(self, inp: dict[str, Any]) -> TaxBiodiversity:
        """
        Invoke the RAG chain synchronously with the given input.

        :param inp: The input data.
        :return: The output of the RAG chain.
        """
        # Prompt/Response logging
        # Use only required keys for formatting to avoid extra keys issues
        prompt_str = self._format_prompt(inp)
        gen = self._build_generation_runnable()
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
            prompt_str = self._format_prompt(inp)
            gen = self._build_generation_runnable()
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
            return self._default_result()

    async def ainvoke_many(
        self,
        inp: dict[str, Any],
        n_samples: int,
        base_seed: int | None = DEFAULT_LLM_SEED,
        seed_mode: str = "incremental",
        temperature: float = 1,
        top_p: float = 1,
        top_logprobs: int = 20,
        use_logprobs: bool = False,
    ) -> dict[str, Any]:
        """Invoke the same prompt multiple times with varied sampling settings."""
        # NOTE: In multi-sampling mode we only request/save token-level logprobs for the
        # base (main) response. Stochastic samples must not request or save logprobs.
        use_logprobs_samples = False
        use_logprobs_base = bool(getattr(self, "_logprobs_path", None))
        prompt_str = self._format_prompt(inp)
        prompt_id = "sha256:" + hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
        rsid = inp.get("RSID")
        prompt_index: int | None = inp.get("prompt_index")
        self._log_multisample_prompt(
            prompt=prompt_str,
            inp=inp,
            prompt_id=prompt_id,
            rsid=rsid,
            prompt_index=prompt_index,
            n_samples=n_samples,
            base_seed=base_seed,
            seed_mode=seed_mode,
            temperature=temperature,
            top_p=top_p,
            use_logprobs=use_logprobs_samples,
            top_logprobs=None,
        )

        def _sample_seed(sample_idx: int) -> int | None:
            if seed_mode == "none":
                return None
            if base_seed is None:
                return None
            return base_seed + sample_idx

        # Base response (deterministic main prediction): same prompt, temperature=0.
        base_seed_used: int | None = None
        if seed_mode != "none":
            base_seed_used = base_seed
        base_temperature = 0.0
        base_top_p = 1.0
        base_result = self._default_result()
        base_response_text = ""
        base_parse_ok = False
        base_parse_error: str | None = None
        try:
            gen_base = self._build_generation_runnable(
                temperature=base_temperature,
                top_p=base_top_p,
                seed=base_seed_used,
                top_logprobs=top_logprobs,
                use_logprobs=use_logprobs_base,
            )
            ai_msg_base = await gen_base.ainvoke(input=inp)
            base_response_text = getattr(ai_msg_base, "content", "") or ""
            base_result = self.output_parser.invoke(base_response_text)
            base_parse_ok = True
            if use_logprobs_base:
                self._log_logprobs(
                    prompt_str,
                    ai_msg_base,
                    rsid=rsid,
                    prompt_id=prompt_id,
                    sample_idx=None,
                    sample_seed=base_seed_used,
                    schema_version="logprob_v2",
                    gen_params_override={
                        "logprobs": True,
                        "temperature": base_temperature,
                        "top_p": base_top_p,
                        "seed": base_seed_used,
                        "top_logprobs": top_logprobs,
                    },
                )
        except Exception as e:
            base_parse_error = f"{type(e).__name__}: {e}"
            logger.error("Error in RAGChainBuilder.ainvoke_many base response:")
            logger.error(f"rsid={rsid}")
            logger.error(traceback.format_exc())

        self._log_multisample_base(
            prompt_id=prompt_id,
            rsid=rsid,
            prompt_index=prompt_index,
            seed_mode=seed_mode,
            seed=base_seed_used,
            temperature=base_temperature,
            top_p=base_top_p,
            response_text=base_response_text,
            response=base_result,
            parse_ok=base_parse_ok,
            parse_error=base_parse_error,
        )

        results: list[TaxBiodiversity] = []
        sample_seeds: list[int | None] = []
        for sample_idx in range(n_samples):
            sample_seed = _sample_seed(sample_idx)
            sample_seeds.append(sample_seed)
            ai_msg: Any | None = None
            response_text = ""
            parse_ok = False
            parse_error: str | None = None
            result = self._default_result()
            try:
                gen = self._build_generation_runnable(
                    temperature=temperature,
                    top_p=top_p,
                    seed=sample_seed,
                    top_logprobs=top_logprobs,
                    use_logprobs=use_logprobs_samples,
                )
                ai_msg = await gen.ainvoke(input=inp)
                response_text = getattr(ai_msg, "content", "") or ""
                result = self.output_parser.invoke(response_text)
                parse_ok = True
            except Exception as e:
                parse_error = f"{type(e).__name__}: {e}"
                logger.error("Error in RAGChainBuilder.ainvoke_many sample:")
                logger.error(f"sample_idx={sample_idx}, rsid={rsid}")
                logger.error(traceback.format_exc())

            self._log_multisample_sample(
                prompt_id=prompt_id,
                prompt_index=prompt_index,
                sample_idx=sample_idx,
                sample_seed=sample_seed,
                rsid=rsid,
                temperature=temperature,
                top_p=top_p,
                response_text=response_text,
                response=result,
                parse_ok=parse_ok,
                parse_error=parse_error,
            )
            # Samples never request/save logprobs in this multi-sampling mode.
            results.append(result)

        return {
            "prompt_id": prompt_id,
            "base_result": base_result,
            "base_seed": base_seed_used,
            "base_temperature": base_temperature,
            "base_top_p": base_top_p,
            "results": results,
            "sample_seeds": sample_seeds,
        }

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

    def _log_multisample_prompt(
        self,
        *,
        prompt: str,
        inp: dict[str, Any],
        prompt_id: str,
        rsid: str | None,
        prompt_index: int | None,
        n_samples: int,
        base_seed: int | None,
        seed_mode: str,
        temperature: float,
        top_p: float,
        use_logprobs: bool,
        top_logprobs: int | None,
    ) -> None:
        """Write one prompt-level record for a multi-sample generation run."""
        if not getattr(self, "_multisample_prompt_path", None):
            return

        try:
            Path(self._multisample_prompt_path).expanduser().parent.mkdir(
                parents=True, exist_ok=True
            )
            record = {
                "prompt_id": prompt_id,
                "rsid": rsid,
                "prompt_index": prompt_index,
                "caption": inp.get("caption", ""),
                "context": inp.get("context", ""),
                "prompt": prompt,
                "sampling": {
                    "num_samples": n_samples,
                    "base_seed": base_seed,
                    "seed_mode": seed_mode,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": use_logprobs,
                    "top_logprobs": top_logprobs,
                },
            }
            with open(self._multisample_prompt_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_err:
            logger.warning(f"Failed to log multisample prompt: {log_err}")

    def _log_multisample_base(
        self,
        *,
        prompt_id: str,
        rsid: str | None,
        prompt_index: int | None,
        seed_mode: str,
        seed: int | None,
        temperature: float,
        top_p: float,
        response_text: str,
        response: Any,
        parse_ok: bool,
        parse_error: str | None,
    ) -> None:
        """Write one base-response record for a multi-sample generation run."""
        if not getattr(self, "_multisample_base_path", None):
            return

        try:
            Path(self._multisample_base_path).expanduser().parent.mkdir(
                parents=True, exist_ok=True
            )
            if hasattr(response, "model_dump"):
                response_obj = response.model_dump()
            else:
                response_obj = str(response)
            classification = (
                response_obj.get("classification", {})
                if isinstance(response_obj, dict)
                else {}
            )
            canonical = "|".join(
                str(classification.get(rank, "N/A"))
                for rank in (
                    "Kingdom",
                    "Phylum",
                    "Class",
                    "Order",
                    "Family",
                    "Genus",
                    "Species",
                )
            )
            response_hash = (
                "sha256:" + hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                if response_text
                else None
            )
            record = {
                "prompt_id": prompt_id,
                "sample_role": "base",
                "rsid": rsid,
                "prompt_index": prompt_index,
                "seed_mode": seed_mode,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": bool(getattr(self, "_logprobs_path", None)),
                "parse_ok": parse_ok,
                "parse_error": parse_error,
                "response_text": response_text,
                "response_text_hash": response_hash,
                "response": response_obj,
                "classification_canonical": canonical,
            }
            with open(self._multisample_base_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_err:
            logger.warning(f"Failed to log multisample base response: {log_err}")

    def _log_multisample_sample(
        self,
        *,
        prompt_id: str,
        prompt_index: int | None,
        sample_idx: int,
        sample_seed: int | None,
        rsid: str | None,
        temperature: float,
        top_p: float,
        response_text: str,
        response: Any,
        parse_ok: bool,
        parse_error: str | None,
    ) -> None:
        """Write one sample-level record for a multi-sample generation run."""
        if not getattr(self, "_multisample_samples_path", None):
            return

        try:
            Path(self._multisample_samples_path).expanduser().parent.mkdir(
                parents=True, exist_ok=True
            )
            if hasattr(response, "model_dump"):
                response_obj = response.model_dump()
            else:
                response_obj = str(response)
            classification = (
                response_obj.get("classification", {})
                if isinstance(response_obj, dict)
                else {}
            )
            canonical = "|".join(
                str(classification.get(rank, "N/A"))
                for rank in (
                    "Kingdom",
                    "Phylum",
                    "Class",
                    "Order",
                    "Family",
                    "Genus",
                    "Species",
                )
            )
            response_hash = (
                "sha256:" + hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                if response_text
                else None
            )
            record = {
                "prompt_id": prompt_id,
                "sample_idx": sample_idx,
                "sample_id": f"{prompt_id}:{sample_idx}",
                "sample_role": "sample",
                "rsid": rsid,
                "prompt_index": prompt_index,
                "sample_seed": sample_seed,
                "temperature": temperature,
                "top_p": top_p,
                "parse_ok": parse_ok,
                "parse_error": parse_error,
                "response_text": response_text,
                "response_text_hash": response_hash,
                "response": response_obj,
                "classification_canonical": canonical,
            }
            with open(self._multisample_samples_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_err:
            logger.warning(f"Failed to log multisample sample: {log_err}")

    def _log_logprobs(
        self,
        prompt: str,
        ai_message: Any,
        rsid: str | None = None,
        *,
        path_override: str | None = None,
        prompt_id: str | None = None,
        sample_idx: int | None = None,
        sample_seed: int | None = None,
        schema_version: str = "logprob_v1",
        gen_params_override: dict[str, Any] | None = None,
    ) -> None:
        """Append a token-level logprobs record to the configured JSONL file (if any)."""
        target_path = path_override or getattr(self, "_logprobs_path", None)
        if not target_path:
            return
        append_logprobs_from_langchain_message(
            target_path=str(target_path),
            prompt=prompt,
            ai_message=ai_message,
            rsid=rsid,
            prompt_id=prompt_id,
            sample_idx=sample_idx,
            sample_seed=sample_seed,
            schema_version=schema_version,
            gen_params_override=gen_params_override,
        )

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
        multisample_prompt_path: str | None = None,
        multisample_base_path: str | None = None,
        multisample_samples_path: str | None = None,
        multisample_logprobs_path: str | None = None,
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
        self.model = RAGChainBuilder(
            self.retriever,
            log_path=log_path,
            logprobs_path=logprobs_path,
            multisample_prompt_path=multisample_prompt_path,
            multisample_base_path=multisample_base_path,
            multisample_samples_path=multisample_samples_path,
            multisample_logprobs_path=multisample_logprobs_path,
        )

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

    async def ainvoke_many(
        self,
        caption: str,
        RSID: str | None = None,
        prompt_index: int | None = None,
        n_samples: int = 1,
        base_seed: int | None = DEFAULT_LLM_SEED,
        seed_mode: str = "incremental",
        temperature: float = 1,
        top_p: float = 1,
        use_logprobs: bool = False,
        top_logprobs: int = 20,
    ) -> dict[str, Any]:
        """Invoke the RAG model multiple times on the same retrieved prompt."""
        docs = await self.aretrieve(caption=caption)
        formatted_docs = format_docs(docs)
        inp: dict[str, Any] = {"context": formatted_docs, "caption": caption}
        if RSID is not None:
            inp["RSID"] = RSID
        if prompt_index is not None:
            inp["prompt_index"] = prompt_index
        return await self.model.ainvoke_many(
            inp=inp,
            n_samples=n_samples,
            base_seed=base_seed,
            seed_mode=seed_mode,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
            # Samples never request/save logprobs in multi-sampling.
            use_logprobs=False,
        )
