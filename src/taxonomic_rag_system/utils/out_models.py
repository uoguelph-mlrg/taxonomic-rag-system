"""
Provides Pydantic models for structuring and validating V/LLM data.

Used for taxonomic classification, biodiversity knowledge, and image captioning.

Classes:
    TaxBiodiversity:
        A model for generating taxonomic classification and biodiversity knowledge.
        Includes ancestral and specific traits, commentary, and relevant biological
        knowledge.

    Tax:
        A model for generating taxonomic classification. Represented as a dictionary
        with keys corresponding to taxonomic ranks from Kingdom to Species.

    MultiQuery:
        A model for constructing multiple queries related to biodiversity to diversify
        queries to a vectorstore.

    Caption:
        A model for structuring the output of a language model captioner. Generates
        detailed image captions of organisms.
"""

from pydantic import BaseModel, Field


class TaxBiodiversity(BaseModel):
    """
    Model for generating taxonomic classification and biodiversity knowledge.

    Taxonomic classification expecting dict[str, str] with keys from Kingdom -> Species.
    """

    classification: dict[str, str] = Field(
        description="""Predicted taxonomic classification dictionary, where keys of dictionary are taxonomic ranks, ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'], and values of dictionary are predicted classification for a given taxonomic rank."""
    )
    ancestral: str = Field(
        description="Describe traits that are seen in the image caption of the new organism and also seen in describing the organisms in the context."
    )
    specific: str = Field(
        description="Describe traits that are seen in the image caption of the new organism, may indicate more unique features within the taxa and not necessarily seen in describing the organisms in the context."
    )
    commentary: str = Field(
        description="""Reasoning or evidence in the context that the caption has been classified the way it has. Point to the ancestral traits and specific traits that support the taxonomic classification. Describe any uncertainty or ambiguity. Describe what would be necessary to make a finer-grained classification (i.e. to the next taxonomic rank)"""
    )
    bio_knowledge: str = Field(
        description="Knowledge about organism relevant to biodiversity"
    )


class Tax(BaseModel):
    """
    Model for generating taxonomic classification.

    Taxonomic classification expecting dict[str, str] with keys from Kingdom -> Species.
    """

    classification: dict[str, str] = Field(
        description="""Predicted taxonomic classification dictionary, where keys of dictionary are taxonomic ranks, ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'], and values of dictionary are predicted classification for a given taxonomic rank.""",
        examples=[
            {
                "Kingdom": "Animalia",
                "Phylum": "Anthropoda",
                "Class": "Insecta",
                "Order": "Arachnida",
                "Family": "Araneae",
                "Genus": "N/A",
                "Species": "N/A",
            },
            {
                "Kingdom": "Animalia",
                "Phylum": "N/A",
                "Class": "N/A",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A",
            },
            {
                "Kingdom": "Animalia",
                "Phylum": "Anthropoda",
                "Class": "Insecta",
                "Order": "Odonata",
                "Family": "Cordulegastridae",
                "Genus": "Anotogaster",
                "Species": "N/A",
            },
        ],
    )


class MultiQuery(BaseModel):
    """Model for constructing multiple queries to the vectorstore."""

    queries: list[str] = Field(
        description="list of generated questions about biodiversity. Each element of the list should be a question that can be asked to taxonomically classify the organism in the caption by accessing documents in the vectorstore. The questions should be related to the image caption and the context of the image. The questions should be diverse and cover different aspects of features, functions, abiotic and biotic relationships.",
    )


class Caption(BaseModel):
    """Pydantic object to be used for structuring LLM captioner output."""

    caption: str = Field(
        description="Detailed image caption containing all visible morphology details of the organism in the image with emphasis on visible features of the organism and mention of the surroundings."
    )


class Chunk(BaseModel):
    """Represents a text chunk with contextual info and a flag indicating usefulness."""

    contextual_text: str = Field(
        description="""
        Text to contextualize the chunk in the document.
        Include the names of taxonomic groups mentioned that may be the subject of the chunk.
        """
    )
    useful: bool = Field(
        description="""
        True if it contains actual descriptive content about organisms or anything relevant for taxonomy. A document with only a citation, header, references, or other non-descriptive text should be marked as not useful (False)
        """
    )
