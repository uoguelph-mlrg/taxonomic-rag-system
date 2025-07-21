"""
Evaluations and dataloaders with the image datasets, such as living arthropods.

It includes a PyTorch Dataset class for loading and processing images of living arthropods
and an evaluator class for batch processing with dataloaders.

Classes
LivingArthropodImageClassDataset
    A PyTorch Dataset for loading and processing images of living arthropods, filtered
    to include only images belonging to the phylum 'Arthropoda'.

LivingArthropodEvaluator
    A class to evaluate living arthropods using a dataset of images, providing functionality
    to initialize the dataset and create a dataloader for batch processing.

Functions
---------
process_row(row)
    Process a single row of a dataset containing image and taxonomic information,
    extracting relevant details and handling errors gracefully.
"""

from typing import Any, Dict, Tuple

from datasets import load_dataset, Image as HFImage, Dataset
from torch.utils.data import DataLoader, Dataset
import sys

from taxonomic_rag_system.utils.living_arthropods_helpers import custom_collate_fn


class LivingArthropodImageClassDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing images of living arthropods.

    Sourced from the HuggingFace dataset "imageomics/rare-species".
    This dataset filters include only images belonging to the phylum 'Arthropoda'.

    Attributes
    ----------
    img_output_pairs (list): A list of tuples where each tuple contains an image
        object and its corresponding classification dictionary (with rare-species ID).

    Methods
    -------
    __len__(): Returns the number of image-class pairs in the dataset.
    __getitem__(idx): Retrieves the image and class dictionary at the specifie
        index.
    """

    def __init__(self) -> None:
        self.img_output_pairs: list[Tuple[Any, Dict[str, Any]]] = []
        self._set_up()

    def _set_up(self) -> None:
        # Load the dataset 
        data_files = "/home/vphung/projects/aip-gwtaylor/vphung/datasets/living-arthropods-dataset/living-arthropod-dataset.csv"
        ds = load_dataset("csv", data_files=data_files, name="arthropod_data")
        ds = ds["train"].select(range(0, 240))
        ds = ds.cast_column("ImageUrl", HFImage())
        example = ds[1]
        
        # Checking the Data to see if there are images
        print(f"Raw example data: {example}")
        print(f"Image field value: {example.get('ImageUrl', 'NO IMAGE FIELD')}")
        
        # Checking process_row to see if images are returning
        try:
            result = process_row(example)
            
            if result is None:
                print("❌ process_row returned None")
            elif result.get('image') is None:
                print("❌❌ Image field is None in result")
            else:
                print("✅ Image successfully loaded")
                print(f"Image type: {type(result['image'])}")
                
                if hasattr(result['image'], 'size'):
                    print(f"Image size: {result['image'].size}")
                    
        except Exception as e:
            print(f"❌ Error in process_row: {e}")
        
        # ISSUE LIES HERE =========================================================
        
        # Map the processing function, filtering out None results
        try:
            processed_ds = ds.map(process_row, batched=False, num_proc=1)
            print("✅ HFImage() approach worked!")
        except Exception as e:
            print(f"❌ HFImage() approach failed: {e}")
            
        #processed_ds = ds.map(process_row)
        
        small_subset = ds.select(range(240))  
        processed_ds = small_subset.map(process_row)
        # =========================================================
        
        #DEBUGGING THE MAP FUNCTION
        print("🔍 DEBUGGING PROCESSED DATASET:")
        for i in range(min(3, len(processed_ds))):  # Check first 3 examples
            example = processed_ds[i]
            print(f"Example {i}:")
            print(f"  - image: {type(example.get('image', 'MISSING'))}")
            print(f"  - class_dict: {example.get('class_dict', 'MISSING')}")
            print(f"  - phylum: {example.get('class_dict', {}).get('Phylum', 'MISSING')}")
            print()
        
        print(f"Pre-filter LENGTH: {processed_ds.num_rows}")
        good_ds = processed_ds.filter(lambda example: example["ImageUrl"] is not None)
        print(f"Post-filter LENGTH: {good_ds.num_rows}")
        
        # Filter the dataset to include only rows where the phylum is 'Arthropoda'
        processed_ds = good_ds.filter(
            lambda row: row["class_dict"]["Phylum"] == "Arthropoda"
        )
        print(f"Arthropod Images: {processed_ds.num_rows}")
        
        # Safety check before accessing elements
        if processed_ds.num_rows == 0:
            raise ValueError("No valid Arthropoda images found in dataset after processing")
        
        if processed_ds.num_rows <= 1:
            raise ValueError(f"Dataset too small: only {processed_ds.num_rows} valid Arthropoda images")
        
        self.img_output_pairs = list(
            zip(processed_ds["ImageUrl"], processed_ds["class_dict"])
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


class LivingArthropodEvaluator:
    """
    A class to build a torch dataloader for evaluation using the living arthropods images.

    This class initializes a dataset of living arthropods images and
    creates a dataloader for batch processing.

    Attributes
    ----------
        dataset (LivingArthropodImageClassDataset):
            The dataset containing images of living arthropods.
    """

    def __init__(self) -> None:
        """Initialize the LivingArthropodEvaluator with a dataset of living arthropods images."""
        self.dataset = LivingArthropodImageClassDataset()

    def dataloader(self, batch_size: int = 16) -> DataLoader:
        """
        Create a dataloader for the living arthropods dataset.

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
            - "LivingArthropod_id": The unique identifier for the living arthropods.
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
                - "RSID": The living arthropods ID or None if an error occurred.
                - "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species":
                  Corresponding taxonomic details or None if an error occurred.
    """
    #TIME TO CHECK 
    print(f"🔍 ENTERING process_row for ID: {row.get('id', 'UNKNOWN')}")
    print(f"🔍 Row type: {type(row)}")
    print(f"🔍 Row keys: {list(row.keys()) if hasattr(row, 'keys') else 'NO KEYS'}")
    
    # Add this check to see if the row structure is different
    if 'ImageUrl' not in row:
        print(f"❌ ImageUrl not found in row! Available keys: {list(row.keys())}")
        raise ValueError("ImageUrl field missing")
    
    print(f"🔍 Processing row ID: {row['id']}")
    print(f"🔍 Input image type: {type(row['ImageUrl'])}")
    print(f"🔍 Input phylum: {row['Phylum']}")
    print(f"🔍 About to access ImageUrl...")

    try:
        
        loaded_image = row["ImageUrl"]
        
        # TIME TO DEBUG
        print(f"🔍 Got image: {type(loaded_image)}")
        
        
        if loaded_image is None:
            print("Image is None")
            raise ValueError("Image is None")
        
        # Check if the image object has the 'getexif' attribute without loading it
        if not hasattr(loaded_image, 'size'):
            print("Image doesn't have size attribute")
            raise ValueError("Invalid image object")
       
        print(f"Returning Image Now {loaded_image}")
        
        return {
            "image": loaded_image,
            "class_dict": {
                "id": row["id"],
                "Kingdom": row["Kingdom"],
                "Phylum": row["Phylum"],
                "Class": row["Class"],
                "Order": row["Order"],
                "Family": row["Family"],
                "Genus": row["Genus"],
                "Species": row["Species"],
                "inat-id": row["inat-id"],
		        "inat-obs-count": row["inat-obs-count"]
            },
        }
    except Exception as e:
        # Log the error
        print(f"Error processing image: {e}\nid: {row['id']}")
        
        # DELETE THIS LATER =================================================================================
        print(f"❌ EXCEPTION in process_row for ID {row.get('id', 'UNKNOWN')}: {e}")
        print(f"❌ Exception type: {type(e)}")
        # Add this line to make sure we see it
        
        print(f"❌ EXCEPTION DETAILS:", file=sys.stderr)

        # Return an empty dict to skip this row
        return {
            "image": None,
            "class_dict": {
                "id": "",
                "Kingdom": "",
                "Phylum": "",
                "Class": "",
                "Order": "",
                "Family": "",
                "Genus": "",
                "Species": "",
                "inat-id": "",
		        "inat-obs-count": "",
            },
        }
