"""
Module providing the "Simple RAG Model" from the `ImageRAGModel` class.

This model integrates image processing, caption generation, and
retrieval-augmented generation (RAG) for taxonomic classification
and biodiversity knowledge extraction. It also includes methods for querying images,
generating captions, and evaluating datasets of living arthropods.

Classes
ImageCaptionModel
    A class that combines image processing, descriptive captioning, and RAG-based
    retrieval to perform taxonomic classification and biodiversity knowledge extraction.
NaiveVLModel
    A VLM for taxonomic classification and biodiversity knowledge extraction.

Dependencies
------------
- asyncio
- gc
- os
- pathlib.Path
- torch
- openai.AsyncOpenAI
- taxonomic_rag_system.utils.evaluator.LivingArthropodEvaluator
- taxonomic_rag_system.utils.helpers.simple_string_output
- taxonomic_rag_system.utils.image_processor.ImageProcessor
- taxonomic_rag_system.utils.out_models.TaxBiodiversity
- taxonomic_rag_system.utils.retriever.WikiStellaRAGModel
- taxonomic_rag_system.utils.vision_models.DescriptiveCaptioner
- taxonomic_rag_system.utils.vision_models.TaxClassifierVLM

Environment Variables
---------------------
- OPENAI_API_KEY: API key for OpenAI services, loaded from `~/.openai.key`.
- OPENROUTER_API_KEY: API key for OpenRouter services, loaded from `~/.openrouter.key`.

This module is designed for asynchronous workflows and requires an appropriate
event loop to execute its methods.
"""

import asyncio
import gc
import os
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import torch
from openai import AsyncOpenAI

# Local imports
from taxonomic_rag_system.utils.living_arthropods_evaluator import (
    LivingArthropodEvaluator,
)
from taxonomic_rag_system.utils.living_arthropods_helpers import simple_string_output
from taxonomic_rag_system.utils.image_processor import ImageProcessor
from taxonomic_rag_system.utils.out_models import TaxBiodiversity
from taxonomic_rag_system.utils.retriever import WikiStellaRAGModel
from taxonomic_rag_system.utils.vision_models import (
    DescriptiveCaptioner,
    TaxClassifierVLM,
)


class _QueryImageOutput(TypedDict):
    caption: str
    results: TaxBiodiversity
    context: str


def load_api_keys() -> None:
    """Load API keys from files."""
    # Set API key env variables w/ `.openai.key` and `.openrouter.key` files in home dir
    with open(Path.home() / ".openai.key", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()
    with open(Path.home() / ".openrouter.key", "r") as f:
        os.environ["OPENROUTER_API_KEY"] = f.read().strip()


class ImageCaptionModel:
    """
    Taxonomically classify with confidence reasoning and biodiversity knowledge.

    This class integrates an image processor and  a descriptive captioner,
    to provide functionalities such as querying images to full pipeline
    or simply generate captions, as well as full evaluation runs over the
    living arthropods dataset.

    Attributes
    ----------
    image_processor : ImageProcessor
        An instance responsible for processing input images into a suitable format.
    captioner : DescriptiveCaptioner
        An instance for generating descriptive captions for images.
    rag_model : WikiStellaRAGModel
        A RAG model instance for retrieving and generating information
        based on captions.

    Methods
    -------
    get_device():
        Retrieve the device (CPU or GPU) on which the RAG model is currently loaded.
    async query_image(
    image_url=None, image_obj=None, image_path=None, context=False, verbose=1
    ):
        Asynchronously queries an image to generate a caption, retrieve results,
        and optionally provide context.
    async caption_image(image_url=None, image_obj=None, image_path=None):
        Caption an image provided via URL, object, or file path.
    async rarespecies_dataset_run(verbose=1):
        Asynchronously processes a dataset of living arthropods images using the RAG system.
    """

    def __init__(
        self,
        cap: Optional[AsyncOpenAI] = None,
        model: str = "gpt-4o", #originally gpt-4o is here
        use_openrouter: bool = True 
    ):
        load_api_keys()
        if cap is None:
            if use_openrouter:
                # configuring AsyncOpenAI for OpenRouter
                cap = AsyncOpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1"
                )
            else:
                cap = AsyncOpenAI()
        
        self.image_processor = ImageProcessor()
        self.captioner = DescriptiveCaptioner(cap=cap, model=model)
        self.tax_classifier = TaxClassifierVLM(model=model, cap=cap) # eventually add it all in one model (generating captions and taxonomically classify for faster baseline experiment)

    async def query_image(
        self,
        image_url: Optional[str] = None,
        image_obj: Optional[Any] = None,
        image_path: Optional[str] = None,
        context: bool = False,
        verbose: int = 1,
    ) -> _QueryImageOutput:
        """
        Asynchronously generates a caption, retrieves, and optionally adds context.

        Args:
            image_url (str, optional): URL of the image to be processed.
                    Defaults to None.
            image_obj (object, optional): Image object to be processed.
                    Defaults to None.
            image_path (str, optional): Local file path of the image to be processed.
                    Defaults to None.
            context (bool, optional): Whether to retrieve additional context documents.
                    Defaults to False.
            verbose (int, optional): Verbosity level for logging.
                    Defaults to 1.

        Returns
        -------
            TypedDict containing the following keys:
                - "caption" (str): Generated caption for the image.
                - "results" (object): Results from the RAG model invocation.
                - "context" (str): String of page content from retrieved docs,
                  if context is True, else empty string.

        Raises
        ------
            Exception: Captures and logs any exceptions that occur during processing,
                       providing a default output structure.
        """
        try:
            image_b64 = self.image_processor.process_image(
                image_url=image_url, image_obj=image_obj, image_path=image_path
            )
            caption = await self.captioner.generate_caption(image_b64) # Caption Generation
            
            if verbose > 0:
                print(f"Caption Generated: {caption}") # Print out caption 
                results = TaxBiodiversity(
                classification={
                    "Kingdom": "Animalia",
                    "Phylum": "N/A",
                    "Class": "N/A",
                    "Order": "N/A",
                    "Family": "N/A",
                    "Genus": "N/A",
                    "Species": "N/A",
                },
                ancestral="",
                specific="",
                commentary="",
                bio_knowledge="",
            )
                
        except Exception as er:
            print(f"{er} occurred")
            caption = ""
            results = TaxBiodiversity(
                classification={
                    "Kingdom": "Animalia",
                    "Phylum": "N/A",
                    "Class": "N/A",
                    "Order": "N/A",
                    "Family": "N/A",
                    "Genus": "N/A",
                    "Species": "N/A",
                },
                ancestral="",
                specific="",
                commentary="",
                bio_knowledge="",
            )
        if not context:
            cntxt = ""
        output = _QueryImageOutput(caption=caption, results=results, context=cntxt)

        if verbose > 0:
            print("queried...")
        return output

    async def caption_image(
        self,
        image_url: Optional[str] = None,
        image_obj: Optional[Any] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """
        Generate a caption for an image provided via URL, object, or file path.

        This asynchronous method processes the input image and generates a textual
        caption describing the content of the image.

        Args:
            image_url (str, optional): The URL of the image to be captioned.
            image_obj (object, optional): An in-memory image object to be captioned.
            image_path (str, optional): The file path to the image to be captioned.

        Returns
        -------
            str: A generated caption describing the content of the image.

        Raises
        ------
            ValueError: If none of (image_url, image_obj, image_path) are provided.
        """
        image_b64 = self.image_processor.process_image(
            image_url=image_url, image_obj=image_obj, image_path=image_path
        )
        return await self.captioner.generate_caption(image_b64)

    async def livingarthropods_dataset_run(self, verbose: int = 1) -> list[dict[str, Any]]:
        """
        Asynchronously processes a dataset of living arthropods images using a RAG system.

        Args:
            verbose (int, optional): Verbosity level for logging and debugging.
                - 0: No output.
                - 1: Minimal output (default).
                - 2: Detailed taxonomy-level comparison.
                - 3: Includes caption and response details.

        Returns
        -------
            list: A list of dictionaries containing the processed output for each batch.
            Each dictionary includes:
                - "caption" (str): Generated caption for the image.
                - "true_class" (dict): Ground truth taxonomy classification w/o "RSID".
                - "context" (str or None): Contextual information, if available.
                - "ancestral" (str): Ancestral classification result.
                - "specific" (str): Specific classification result.
                - "commentary" (str): Commentary on the classification.
                - "biodiversity" (str): Biodiversity-related knowledge.
                - "guess_class" (dict): Predicted taxonomy classification w/o "Domain".
                - "response" (str): Simplified string representation of the output.
                - "RSID" (str): Unique identifier for the living arthropods.

        Notes
        -----
            - Taxonomy levels - "Phylum", "Class", "Order", "Family", "Genus", "Species"
            - Verbose levels >1 provide detailed logging for debugging purposes.
        """
        dataloader = LivingArthropodEvaluator().dataloader()
        
        # Processing loop for batch of images
        outputs = []
        
        for image_objs, class_dicts in dataloader:
            tasks, true_classes, batch = [], [], []
            
            # Classify on each caption
            for img_obj, class_dict in zip(image_objs, class_dicts):
                tasks.append(self.query_image(image_obj=img_obj, context=False))
                true_classes.append(class_dict)
            vlm_responses = await asyncio.gather(*tasks)
            torch.cuda.empty_cache()
            
            for i, response in enumerate(vlm_responses):
                true_class = {
                    level: true_classes[i][level]
                    for level in true_classes[i]
                    if level != "id"
                }
                # Output parse
                cls = response
                guess_class = {
                    level: cls[level] for level in cls if cls[level] != "N/A"
                }
                if verbose > 1:
                    taxonomy_levels = [
                        "Phylum",
                        "Class",
                        "Order",
                        "Family",
                        "Genus",
                        "Species",
                    ]
                    print(f"{'Rank':<10}{'True':<30}{'Pred':<30}")
                    print("=" * 50)
                    for level in taxonomy_levels:
                        true_value = true_class.get(level, "")
                        pred_value = guess_class.get(level, "")
                        print(f"{level:<10}{true_value:<30}{pred_value:<30}")
                    print("=" * 50)
                output: dict[str, Union[str, dict[str, str]]] = {
                    "true_class": true_class,
                    "guess_class": guess_class,
                    "id": true_classes[i]["id"],
                }
                batch.append(output)
            outputs.extend(batch)
        gc.collect()
        torch.cuda.empty_cache()
        return outputs


class NaiveVLModel:
    """
    A class for tasking a VLM with taxonomic classification.

    Class methods for evaluation over the living arthropods dataset.

    Attributes
    ----------
        image_processor (ImageProcessor): For image loading and processing.
        vlm (TaxClassifierVLM): For captioning and classification.

    Methods
    -------
        __init__(model="google/gemini-2.0-flash-001"):
            Initializes the NaiveVLModel with a specified (OpenRouter) VLM model.

        query(image_path=None, image_obj=None):
            Processe an image and generate a classification guess using the VLM.

        async livingarthropods_dataset_run(verbose=1):
            Evaluate on the rare-species dataset then show, write and return results.
    """

    """"""

    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-001",
        openrouter: bool = True,
    ):
        load_api_keys()
        cap = AsyncOpenAI() if not openrouter else None
        self.image_processor = ImageProcessor()
        self.model = TaxClassifierVLM(model=model, cap=cap)

    async def query(
        self, image_path: Optional[str] = None, image_obj: Optional[Any] = None
    ) -> dict[str, str]:
        """
        Query the VLM to generate a classification guess for an image.

        Args: (both default to None)
            image_path (str, optional): The image file path to be processed.
            image_obj (object, optional): An in-memory image object to be processed.

        Returns
        -------
            dict: A dictionary containing the classification guess with taxonomic ranks
                  (e.g., Kingdom, Phylum, Class, Order, Family, Genus, Species).
                  If an error occurs during processing,
                  a default dictionary with "N/A" values is returned.
        """
        try:
            image_b64 = self.image_processor.process_image(
                image_path=image_path, image_obj=image_obj
            )
            print("querying...")
            guess_class = await self.model.generate_taxonomy(image_b64)
            guess_class = {
                level: guess_class[level] for level in guess_class if level != "Domain"
            }
        except Exception as er:
            print(f"{er} occurred")
            guess_class = {
                "Kingdom": "Animalia",
                "Phylum": "N/A",
                "Class": "N/A",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A",
            }
        print("queried...")
        return guess_class

    async def livingarthropods_dataset_run(
        self, verbose: int = 1
    ) -> list[dict[str, Union[str, dict[str, Union[str, Any]]]]]:
        """
        Pass over the living-arthropods dataset.

        Args:
            verbose (int, optional): Verbosity level for logging. Defaults to 1.
                - If `verbose > 1`, detailed taxonomy level comparisons will be printed.

        Returns
        -------
            list: A list of dictionaries containing the following keys:
                - "true_class": A dict of true taxonomy levels (excluding "RSID").
                - "guess_class": A dict of predicted taxonomy levels.
                - "RSID": The unique identifier for the living arthropods.
        """
        dataloader = LivingArthropodEvaluator().dataloader()
        
        # Processing loop for batch of images
        outputs = []
        for image_objs, class_dicts in dataloader:
            tasks, true_classes, batch = [], [], []
            # Naive VLM Tax classifier on each image
            for img_obj, class_dict in zip(image_objs, class_dicts):
                tasks.append(self.query(image_obj=img_obj))
                true_classes.append(class_dict)
            # Run the VLM queries in parallel
            vlm_responses = await asyncio.gather(*tasks)
            torch.cuda.empty_cache()
            
            for i, response in enumerate(vlm_responses):
                true_class = {
                    level: true_classes[i][level]
                    for level in true_classes[i]
                    if level != "id"
                }
                # Output parse
                cls = response
                guess_class = {
                    level: cls[level] for level in cls if cls[level] != "N/A"
                }
                if verbose > 1:
                    taxonomy_levels = [
                        "Phylum",
                        "Class",
                        "Order",
                        "Family",
                        "Genus",
                        "Species",
                    ]
                    print(f"{'Rank':<10}{'True':<30}{'Pred':<30}")
                    print("=" * 50)
                    for level in taxonomy_levels:
                        true_value = true_class.get(level, "")
                        pred_value = guess_class.get(level, "")
                        print(f"{level:<10}{true_value:<30}{pred_value:<30}")
                    print("=" * 50)
                output: dict[str, Union[str, dict[str, str]]] = {
                    "true_class": true_class,
                    "guess_class": guess_class,
                    "id": true_classes[i]["id"],
                }
                batch.append(output)
            outputs.extend(batch)
        gc.collect()
        torch.cuda.empty_cache()
        return outputs
