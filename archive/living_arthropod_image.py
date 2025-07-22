"""
Module providing the "Simple RAG Model" from the `ImageRAGModel` class.

This model integrates image processing, caption generation, and or taxonomic classification
and biodiversity knowledge extraction before and after training the model via GRPO.
It also includes methods for querying images, generating captions, and evaluating datasets of living arthropods.

Classes
ImageRAGModel
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
    with open(Path.home() / ".openrouter.key", "r") as f: #using second key.
        os.environ["OPENROUTER_API_KEY"] = f.read().strip()

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

        async rarespecies_dataset_run(verbose=1):
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
            
            guess_class = await self.model.generate_taxonomy(image_b64) # taken from TaxClassiferVLM
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
        self, verbose: int = 1, delay_between_requests: float = 3.0
    ) -> list[dict[str, Union[str, dict[str, Union[str, Any]]]]]:
        """
        Pass over the living arthropods dataset.

        Args:
            verbose (int, optional): Verbosity level for logging. Defaults to 1.
                - If `verbose > 1`, detailed taxonomy level comparisons will be printed.

        Returns
        -------
            list: A list of dictionaries containing the following keys:
                - "true_class": A dict of true taxonomy levels (excluding "RSID").
                - "guess_class": A dict of predicted taxonomy levels.
                - "id": The unique identifier for the living arthropods.
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
            rag_responses = await asyncio.gather(*tasks)
            
            torch.cuda.empty_cache()
            for i, response in enumerate(rag_responses):
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