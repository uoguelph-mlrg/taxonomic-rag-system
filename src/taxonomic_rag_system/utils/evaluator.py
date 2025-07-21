"""
Evaluations and dataloaders with the image datasets, including imageomic's rare species.

It includes a PyTorch Dataset class for loading and processing images of rare species
and an evaluator class for batch processing with dataloaders.

Classes
RareSpeciesImageClassDataset
    A PyTorch Dataset for loading and processing images of rare species, filtered
    to include only images belonging to the phylum 'Arthropoda'.

RareSpeciesEvaluator
    A class to evaluate rare species using a dataset of images, providing functionality
    to initialize the dataset and create a dataloader for batch processing.

Functions
---------
process_row(row)
    Process a single row of a dataset containing image and taxonomic information,
    extracting relevant details and handling errors gracefully.
"""

from typing import Any, Dict, Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from taxonomic_rag_system.utils.helpers import custom_collate_fn


class RareSpeciesImageClassDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing images of rare species.

    Sourced from the HuggingFace dataset "imageomics/rare-species".
    This dataset filters include only images belonging to the phylum 'Arthropoda'.

    Attributes
    ----------
    img_output_pairs (list): A list of tuples where each tuple contains an image
        object and its corresponding classification dictionary (with rare-species ID).

    Methods
    -------
    __len__(): Returns the number of image-class pairs in the dataset.
    __getitem__(idx): Retrieves the image and class dictionary at the specified
        index.
    """

    def __init__(self) -> None:
        self.img_output_pairs: list[Tuple[Any, Dict[str, Any]]] = []
        self._set_up()

    def _set_up(self) -> None:
        # Load the dataset from HuggingFace
        ds = load_dataset("imageomics/rare-species")
        ds = ds["train"].select(range(0, 10))
        # Map the processing function, filtering out None results
        processed_ds = ds.map(process_row)
        print(f"Pre-filter LENGTH: {processed_ds.num_rows}")
        good_ds = processed_ds.filter(lambda example: example["image"] is not None)
        print(f"Post-filter LENGTH: {good_ds.num_rows}")
        # Filter the dataset to include only rows where the phylum is 'Arthropoda'
        processed_ds = good_ds.filter(
            lambda row: row["class_dict"]["Phylum"] == "Arthropoda"
        )
        print(f"Arthropod Images: {processed_ds.num_rows}")
        self.img_output_pairs = list(
            zip(processed_ds["image"], processed_ds["class_dict"])
        )

    def __len__(self) -> int:
        """
        Return the number of image-output pairs in the evaluator.

        Returns
        -------
            int: The number of image-output pairs.
        """
        return len(self.img_output_pairs)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Retrieve the image object and class dictionary at the specified index.

        Args:
            idx (int): The index of the desired item in the img_output_pairs list.

        Returns
        -------
            tuple: A tuple containing the image object and its corresponding class dict.
        """
        image_obj, class_dict = self.img_output_pairs[idx]
        return image_obj, class_dict


class RareSpeciesEvaluator:
    """
    A class to build a torch dataloader for evaluation using the rare species images.

    This class initializes a dataset of rare species images and
    creates a dataloader for batch processing.

    Attributes
    ----------
        dataset (RareSpeciesImageClassDataset):
            The dataset containing images of rare species.
    """

    def __init__(self) -> None:
        """Initialize the RareSpeciesEvaluator with a dataset of rare species images."""
        self.dataset = RareSpeciesImageClassDataset()

    def dataloader(self, batch_size: int = 16) -> DataLoader:
        """
        Create a dataloader for the rare species dataset.

        Args:
            batch_size (int): The number of samples per batch. Defaults to 16.

        Returns
        -------
            DataLoader: A DataLoader instance for iterating over the dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )


def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single row of a dataset containing image and taxonomic information.

    This function checks if the image is loaded well by checking if the image object
    has the 'getexif' attribute and extracts taxonomic classification from the row.
    If a processing error occurs, it logs the error and returns a default empty dict.

    Args:
        row (dict): A dictionary representing a single row of the dataset. It must
            contain the following keys:
            - "image": An image object.
            - "rarespecies_id": The unique identifier for the rare species.
            - "kingdom": The taxonomic kingdom of the species.
            - "phylum": The taxonomic phylum of the species.
            - "class": The taxonomic class of the species.
            - "order": The taxonomic order of the species.
            - "family": The taxonomic family of the species.
            - "genus": The taxonomic genus of the species.
            - "sciName": The scientific name (species) of the organism.

    Returns
    -------
        dict: A dictionary containing:
            - "image": The original image object if valid, otherwise None.
            - "class_dict": A dictionary with taxonomic classification details:
                - "RSID": The rare species ID or None if an error occurred.
                - "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species":
                  Corresponding taxonomic details or None if an error occurred.
    """
    try:
        # Check if the image object has the 'getexif' attribute without loading it
        if not hasattr(row["file_name"], "getexif"):
            raise AttributeError

        return {
            "image": row["file_name"],
            "class_dict": {
                "RSID": row["rarespecies_id"],
                "Kingdom": row["kingdom"],
                "Phylum": row["phylum"],
                "Class": row["class"],
                "Order": row["order"],
                "Family": row["family"],
                "Genus": row["genus"],
                "Species": row["sciName"],
            },
        }
    except Exception as e:
        # Log the error
        print(f"Error processing image: {e}\nRSID: {row['rarespecies_id']}")

        # Return an empty dict to skip this row
        return {
            "image": None,
            "class_dict": {
                "RSID": "",
                "Kingdom": "",
                "Phylum": "",
                "Class": "",
                "Order": "",
                "Family": "",
                "Genus": "",
                "Species": "",
            },
        }
