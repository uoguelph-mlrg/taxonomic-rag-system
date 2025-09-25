"""
Module to process and contextualize text chunks from documents.

Primarily for taxonomic purposes. It includes utilities for:

1. Loading documents from a directory or custom loader.
2. Splitting documents into smaller chunks.
3. Contextualizing and categorizing chunks using an AI model.
4. Filtering out already processed documents.
5. Logging examples of useful and non-useful chunks.

Classes:
    - Chunk: Represents a text chunk with contextual information and a usefulness flag.
    - FilteringCustomLoader: Custom loader for filtering already processed documents.

Functions:
    - _load_up(path): Loads documents from a directory using DirectoryLoader.
    - contextual_retrieval_load(docs, client, source): Processes documents to generate
      useful and non-useful chunks, saving useful chunks to files.
    - pretty_print_chunks(use, notuse): Logs examples of useful and non-useful chunks.
    - _filter_present(): Identifies already processed documents to exclude.
    - contextualize(chunk, doc, client): Asynchronously contextualizes a text chunk
      within the context of a larger document.
    - _filtered_load_up(path): Loads documents using FilteringCustomLoader, excluding
      already processed ones.
    - main(): Main entry point for the script, handling argument parsing and execution.

Usage:
```bash
python chunk_vectorize.py \
    --source <source_directory> \
    --output_path <output_directory> \
    --pers_dir <persistence_directory> \
    --contextualize --write
```
"""

import argparse
import asyncio
import json
import logging
import os
import re
from typing import Iterator, Optional, Tuple

import backoff
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from openai import AsyncOpenAI
from openai.error import APIError, RateLimitError

from taxonomic_rag_system.utils.helpers import load_api_keys
from taxonomic_rag_system.utils.out_models import Chunk
from taxonomic_rag_system.utils.retriever import SafeHuggingFaceEmbeddings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_up(path: str) -> list[Document]:
    docs = []
    loader = DirectoryLoader(
        path, glob="*.txt", recursive=True, use_multithreading=True
    )
    docs.extend(loader.load())
    logger.info("Documents loaded with DirectoryLoader...")
    logger.info(f"{len(docs)} Documents")
    return docs


async def contextual_retrieval_load(
    docs: list[Document],
    client: AsyncOpenAI,
    output_path: str,
    source: str = "",
    write: bool = True,
) -> Tuple[list[Document], list[Document]]:
    """
    Chunk, contextualize and categorize document lists.

    Useful chunks are saved to file.

    Args:
        docs (list): A list of documents to process. list[dict[str, str| dict[str,str]]]
                     Each containing at least a "page_content" key and
                     "metadata" key with  "source" field.
        client (object): A client object used for contextualizing the chunks.
        source (str): A string identifier for the source of the documents,
                      used in naming output files.

    Returns
    -------
        tuple: A tuple containing two lists:
               - useful_chunks (list): Useful chunks after contextualization.
               - notuseful_chunks (list): Not useful chunks after contextualization.
    """
    if source:
        source = source + "_"

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t", ". ", " ", ""],
        chunk_size=768,
        chunk_overlap=100,
    )

    dox = []
    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        dok = dict(doc)
        dok["chunks"] = chunks
        dox.append(dok)

    useful_chunks, notuseful_chunks = [], []
    chunk_total = 0
    for dock in dox:
        filename = dock["metadata"]["source"].split("/")[-1].replace(".txt", "")
        for i, chunk in enumerate(dock["chunks"]):
            response = await contextualize(
                chunk.page_content, dock["page_content"], client
            )
            chunk_total += 1
            chunk.page_content = f"{chunk.page_content}\n\n{response.contextual_text}"
            if response.useful:
                useful_chunks.append(chunk)
                outname = f"{output_path}{source}doc_{filename}_chunk_{i}.txt"
                if write:
                    with open(outname, "w") as f:
                        f.write(chunk.model_dump_json())
                    logger.info(f"Saved to file {filename}")
            else:
                notuseful_chunks.append(chunk)
    logger.info(f"Chunks refined from {chunk_total} to {len(useful_chunks)}")
    return useful_chunks, notuseful_chunks


async def no_contextual_retrieval_load(
    docs: list[Document],
    output_path: str,
    write: bool = True,
) -> list[Document]:
    """
    Chunk, contextualize and categorize document lists.

    Useful chunks are saved to file.

    Args:
        docs (list): A list of documents to process. list[dict[str, str| dict[str,str]]]
                     Each containing at least a "page_content" key and
                     "metadata" key with  "source" field.
        source (str): A string identifier for the source of the documents,
                      used in naming output files.

    Returns
    -------
        chunks: A list of chunks after processing docs.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t", ". ", " ", ""],
        chunk_size=1024,
        chunk_overlap=100,
    )

    dox = []
    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        dok = dict(doc)
        dok["chunks"] = chunks
        dox.append(dok)

    useful_chunks = []
    chunk_total = 0
    for dock in dox:
        filename = dock["metadata"]["source"].split("/")[-1].replace(".txt", "")
        for i, chunk in enumerate(dock["chunks"]):
            chunk_total += 1
            useful_chunks.append(chunk)
            outname = f"{output_path}doc_{filename}_chunk_{i}.txt"
            if write:
                with open(outname, "w") as f:
                    f.write(chunk.model_dump_json())
                logger.info(f"Saved to file {outname}")
    logger.info(f"Chunks refined from {chunk_total} to {len(useful_chunks)}")
    return useful_chunks


def pretty_print_chunks(use: list[Document], notuse: list[Document]) -> None:
    """Log examples of useful and non-useful chunks from contextual embeddings."""
    for chunk in use[0:10]:
        logger.info("=" * 100)
        logger.info(
            f"Example of useful chunk with contextual embedding:\n\n{chunk.page_content}"
        )
        logger.info("=" * 100)
    for chunk in notuse[0:4]:
        logger.info("=" * 100)
        logger.info(
            f"Example of non-useful chunk with reason for exclusion:\n\n{chunk.page_content}"
        )
        logger.info("=" * 100)


def _check_wiki_metadata(full_path: str) -> dict[str, str]:
    tree = full_path.split("/")
    if "Wiki" in tree:
        # Webscrape files looks like Wiki/{source}/<rank>/<tax_name>_{source}.txt
        # where source is Wikipedia, Wikispecies, ToL
        rank = str(tree[-2])
        source = str(tree[-3])
        tax_name = str(tree[-1]).replace(f"_{source}.txt", "")
    else:
        rank = "N/A"
        tax_name = "N/A"

    return {"rank": rank, "tax_name": tax_name, "source": full_path}


class FilteringCustomLoader(BaseLoader):
    """
    A custom loader class for filtering and lazily loading text documents from path.

    This class iterates through a directory, filtering out files already present and
    yields `Document` objects containing the content and metadata of the files.

    Attributes
    ----------
        file_path (str): The root directory path containing the files to be loaded.

    Methods
    -------
        lazy_load() -> Iterator[Document]:
            Lazily loads and yields `Document` objects of files that pass filter.
    """

    def __init__(self, file_path: str, output_path: Optional[str] = None):
        self.file_path = file_path
        if output_path is None:
            output_path = self.file_path
        self.output_path = output_path + "output_chunks/"

    def lazy_load(self) -> Iterator[Document]:
        """
        Recursively loads documents from a specified path, yielding `Document` objects.

        Skips files that are present in the negate filter or have encoding issues.

        Yields
        ------
            Iterator[Document]: An iterator of `Document` objects containing the content
            and metadata of each valid text file.

        Raises
        ------
            UnicodeDecodeError: If a file cannot be read due to encoding issues, it is
            skipped, and a warning is logged.
        """
        negate = _filter_present(output_path=self.output_path)
        for root, _, files in os.walk(self.file_path):
            for file in files:
                if ".txt" in file and file not in negate:
                    full_path = os.path.join(root, file)
                    metadata = _check_wiki_metadata(full_path=full_path)
                    try:
                        with open(full_path, encoding="utf-8") as f:
                            yield Document(
                                page_content=f.read(),
                                metadata=metadata,
                            )
                    except UnicodeDecodeError:
                        logger.warning(
                            f"Skipping file due to encoding issue: {full_path}"
                        )


def _filter_present(output_path: str) -> list[str]:
    negate = []
    for file in os.listdir(output_path):
        if "chunk" in file.split("_"):
            # Regex to extract the filename
            match = re.search(r"doc_(.*?)_chunk_", file)
            if match:
                filename = str(match.group(1))
                file_name = filename + ".txt"
                negate.append(file_name)
    negate = list(set(negate))
    logger.info(f"{len(negate)} docs negated")

    return negate


@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=3)
async def contextualize(chunk: str, doc: str, client: AsyncOpenAI) -> Chunk:
    """
    Asynchronously contextualize a chunk of text within context of larger doc.

    Determine its usefulness for taxonomic purposes and provide additional context.

    Args:
        chunk (str): The chunk of text to be analyzed and contextualized.
        doc (str): The larger document from which the chunk originates.
        client (AsyncOpenAI): The async client for the LLM.

    Returns
    -------
        Chunk: A Pydantic object containing:
            - useful (bool): Indicates if chunk contains descriptive content.
            - contextual_text (str): If useful, provides contextualization text;
              otherwise, explains the type of non-descriptive content.

    Raises
    ------
        AssertionError: If the response is not an instance of the Chunk class.
    """
    system_prompt = """
    You are an expert AI assistant to a taxonomist. You can determine if a user-provided chunk of textual context is useful for a taxonomist and if it is, contextualize it for retrieval in the context of a larger document.

    You are analyzing text from documents written mostly by naturalist throughout history, describing their travels as well as organisms they encounter.

    For each chunk:
    1. First determine if the chunk (alone) contains actual descriptive content about any organisms. A chunk with no mention of organisms, or containing just a citation, header, references, or other non-descriptive text should be marked as not useful (false).
    2. If the chunk contains descriptive content about organisms (usefulness = true), provide a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Also include any taxonomic classification from the document.
    3. If the chunk does NOT contain descriptive content (usefulness = false), briefly state what type of content it contains instead (e.g. "citation", "references header", etc).

    You do not need to give an overview starting statement such as "This document provides a detailed description of the phylum Chordata" but rather, get right into what is being said about Chordata in the chunk.

    Format your response as a Pydantic object with:
    - useful: boolean indicating if descriptive organism content is present
    - contextual_text: string containing either the descriptive contextualization or content type explanation
    """

    document_context_prompt = """
    <document>
    {doc_content}
    </document>
    """

    chunk_context_prompt = """
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>
    """
    # Pass each chunk with its doc of origin to the LLM and parse as Chunk class
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": document_context_prompt.format(doc_content=doc),
                    },
                    {
                        "type": "text",
                        "text": chunk_context_prompt.format(chunk_content=chunk),
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    # Log the raw response for inspection
    logging.debug(f"Raw response: {resp}")

    # Extract the JSON string
    json_content = resp.choices[0].message.content

    # Parse the JSON string to a Python dictionary
    parsed_data = json.loads(json_content)

    # Manually parse the response to a Chunk object
    chunk = Chunk.model_validate(parsed_data)
    assert isinstance(chunk, Chunk)
    return chunk


def _filtered_load_up(path: str, output_path: Optional[str] = None) -> list[Document]:
    docs: list[Document] = []
    loader = FilteringCustomLoader(file_path=path, output_path=output_path or "")
    docs.extend(loader.lazy_load())
    logger.info(
        "Documents loaded with Custom Loader, not including those already created, moving to chunking..."
    )
    logger.info(f"{len(docs)} documents leftover")
    return docs


def build_vectorstore_from(
    docs: list[Document],
    persistence_dir: str,
    embedding_model: str = "dunzhang/stella_en_1.5B_v5",
    collection_name: str = "Wiki_contexted",
    verbose: bool = True,
) -> None:
    """
    Build or update a vectorstore from a list of documents using an embedding model.

    Args:
        docs (list[Document]): A list of docs to be embedded and added to vectorstore.
        persistence_dir (str): The directory where the vectorstore will be persisted.
        embedding_model (str, optional): Defaults to "dunzhang/stella_en_1.5B_v5".
        collection_name (str, optional): Defaults to "Wiki_contexted".
        verbose (bool, optional): Display progress and debug info. Defaults to True.

    Raises
    ------
        ValueError: If there is an issue with loading or creating the vectorstore.

    Notes
    -----
        - If specified collection already exists, add the provided documents to it.
        - If collection does not exist, a new vectorstore created with provided docs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Import embeddings model
    encode_kwargs = {
        # Use faster dot-product instead of cosine sim
        "normalize_embeddings": True,
        "batch_size": 128,
    }
    embeddings = SafeHuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs,
        show_progress=verbose,
    )
    print(f"Building vectorstore from {len(docs)} chunks")

    try:
        # Attempt to load the collection
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persistence_dir,
            embedding_function=embeddings,
        )
        print(f"Collection '{collection_name}' exists. Adding documents to it.")
        vectorstore.add_documents(docs)
    except ValueError:
        # If the collection does not exist, create a new one
        print(
            f"Collection '{collection_name}' does not exist. Creating a new vectorstore."
        )
        vectorstore = Chroma.from_documents(
            docs,
            embedding_function=embeddings,
            collection_name=collection_name,
            persist_directory=persistence_dir,
        )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including source directory, output path, and flags for
        contextualization and writing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        help="Source directory of the documents to be contextualized and used for building vectorstore.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path where contextualized documents will be saved.",
    )
    parser.add_argument(
        "--pers_dir",
        type=str,
        help="Persistence directory for the vectorstore.",
    )
    parser.add_argument(
        "--contextualize",
        action="store_true",
        type=bool,
        help="Flag to indicate whether to contextualize documents.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        type=bool,
        help="Flag to indicate whether to write the contextualized documents.",
    )

    return parser.parse_args()


async def main() -> None:
    """
    Preprocess and/or contextualize document chunks, then build vectorstore from them.

    This function performs the following steps:
    1. Reads the OpenAI API key from a local file and sets it in the environment.
    2. Parses command-line arguments to determine input source, output path,
        persistence dir, and whether to contextualize the document chunks.
    3. Loads documents from the specified source.
    4. Optionally contextualizes the documents using a language model client.
    5. Builds a vector store from the processed document chunks and saves it.

    Arguments:
         None

    Returns
    -------
         None
    """
    args = parse_arguments()
    source = args.source
    output_path = args.output_path
    pers_dir = args.pers_dir
    contextualize = args.contextualize
    write = args.write

    load_api_keys()
    llm = AsyncOpenAI()

    documents = _filtered_load_up(path=source, output_path=output_path)
    if contextualize:
        use, notuse = await contextual_retrieval_load(
            documents,
            llm,
            source=source,
            output_path=output_path,
            write=write,
        )
        logger.info("Finished contextualizing documents.")
        pretty_print_chunks(use, notuse)
    else:
        logger.info("Skipping contextualization step.")
        use = await no_contextual_retrieval_load(
            documents, output_path=output_path, write=write
        )
    build_vectorstore_from(
        use,
        persistence_dir=pers_dir,
        embedding_model="dunzhang/stella_en_1.5B_v5",
        collection_name=f"Wiki_{['un', ''][contextualize]}contexted",
    )
    logger.info("Finished loading chunks into vectorstore.")


if __name__ == "__main__":
    asyncio.run(main())
