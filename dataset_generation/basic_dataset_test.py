#!/usr/bin/env python
"""
Basic Dataset Test - A minimal test of the data generation pipeline using bespokelabs curator
and dataset dimensions to generate synthetic training examples.
"""

import os
import random
import json
import sys
from dataclasses import asdict
from typing import List, Dict, Any

from bespokelabs import curator
from pydantic import BaseModel, Field
from rich import print
from rich.console import Console

from dataset_dimensions import (
    get_all_dimensions, DatasetDimensionValue, DatasetExampleDimensions,
    ContentFocusDimension, ContentFocusDimensionValue,
    FormatDimension, FormatDimensionValue,
    DomainEmphasisDimension, DomainEmphasisDimensionValue,
    ReasoningDimension, ReasoningDimensionValue,
    ConceptualComplexityDimension, ConceptualComplexityDimensionValue,
    AmbiguityLevelDimension, AmbiguityLevelDimensionValue,
    FactualDensityDimension, FactualDensityDimensionValue,
    TemporalFocusDimension, TemporalFocusDimensionValue,
    EmotionalValenceStyleDimension, EmotionalValenceStyleDimensionValue
)

# ENVs
os.environ["TELEMETRY_ENABLED"]="0"

# Constants
MAX_EXAMPLES = 5  # Generate at most 5 examples for testing
MAX_SEQ_LENGTH = 1024  # Max token length (consistent with model config)
OUTPUT_DIR = "dataset_generation/test_output"
RATE_LIMIT_RPM = 60  # Requests per minute
RATE_LIMIT_TPM = 80000  # Tokens per minute


class DimensionSampler:
    """Utility class for sampling from dataset dimensions based on defined weights."""
    
    def __init__(self):
        # Load all dimensions
        self.all_dimensions = get_all_dimensions()
        
        # Create mappings for convenience
        self.dim_name_to_instance = {dim.name: dim for dim in self.all_dimensions}
        self.dim_name_to_value_class = {
            "Content Focus": ContentFocusDimensionValue,
            "Format & Structure": FormatDimensionValue,
            "Domain Emphasis": DomainEmphasisDimensionValue,
            "Reasoning Explicitness": ReasoningDimensionValue,
            "Conceptual Complexity": ConceptualComplexityDimensionValue,
            "Ambiguity Level": AmbiguityLevelDimensionValue,
            "Factual Density": FactualDensityDimensionValue,
            "Temporal Focus": TemporalFocusDimensionValue,
            "Emotional Valence & Style": EmotionalValenceStyleDimensionValue,
        }
        self.dimension_value_to_field_name = {
            "Content Focus": "content_focus",
            "Format & Structure": "format",
            "Domain Emphasis": "domain_emphasis",
            "Reasoning Explicitness": "reasoning",
            "Conceptual Complexity": "conceptual_complexity",
            "Ambiguity Level": "ambiguity_level",
            "Factual Density": "factual_density",
            "Temporal Focus": "temporal_focus",
            "Emotional Valence & Style": "emotional_valence_style",
        }
    
    def sample_example_dimensions(self) -> DatasetExampleDimensions:
        """Sample a complete set of dimension values for a single example."""
        dimension_values_for_example = {}
        
        for dim_name, dimension in self.dim_name_to_instance.items():
            choices = dimension.choices
            weights = [choice.weight for choice in choices]
            # Perform weighted random sampling
            selected_choice = random.choices(choices, weights=weights, k=1)[0]
            
            # Create the specific DimensionValue instance
            ValueClass = self.dim_name_to_value_class[dim_name]
            dimension_value = ValueClass(choice=selected_choice, dimension=dimension)
            
            # Get the field name for the DatasetExampleDimensions dataclass
            field_name = self.dimension_value_to_field_name[dim_name]
            dimension_values_for_example[field_name] = dimension_value
        
        # Create the example dimensions container
        return DatasetExampleDimensions(**dimension_values_for_example)


class Example(BaseModel):
    title: str = Field(description="The title of the example.")
    content: str = Field(description="The generated text content.")

class ContentGenerator(curator.LLM):
    """
    Curator-based content generator implementing the BespokeLabs LLM pattern.
    
    This class handles the generation of training examples using the sampled
    dimensions to construct prompts for the LLM, using LiteLLM as the backend.
    """
    
    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            response_format=Example,
            backend="litellm"
        )
        self.model_name = model_name
        self.max_tokens = MAX_SEQ_LENGTH
        self.temperature = 0.7
    
    def prompt(self, example_data: Dict):
        """
        Create a prompt for the LLM based on the dimension values.
        
        Args:
            example_data: Dictionary containing 'dimensions' (DatasetExampleDimensions) and 'topic' (str)
            
        Returns:
            Tuple of (system_prompt, user_prompt) for the LLM
        """
        dimensions = example_data['dimensions']
        topic = example_data['topic']
        
        system_prompt = (
            "You are an expert data generator tasked with creating high-quality pre-training examples for a large language model. "
            "Your goal is to produce text that exemplifies specific educational and reasoning characteristics."
        )
        
        user_prompt = "Your example must adhere to each of the following 9 orthogonal information dimensions as described below. "
        user_prompt += "Each dimension represents an informational quality that the text can have, independent of the other information dimensions.\n\n"
        
        # Add each dimension with its selected choice
        for field_name, dimension_value in dimensions.items():
            choice = dimension_value['choice']
            dimension = dimension_value['dimension']
            
            user_prompt += f"# {dimension['name']}\n"
            user_prompt += f"Dimension Description: {dimension['description']}\n"
            user_prompt += f"Selected Value: {choice['value']}\n"
            user_prompt += f"Value Meaning: {choice['description']}\n"
            user_prompt += f"Value Generation Instructions: {choice['instruction']}\n"
            user_prompt += f"Value Training Goal: {choice['goal']}\n\n"
        
        # Add topic and length constraint
        user_prompt += f"Topic/Instruction: {topic}\n\n"
        user_prompt += f"Length: Do not produce an answer that is longer than {int(MAX_SEQ_LENGTH * 0.65)} words.\n\n"
        user_prompt += "Ensure the output directly reflects the specified dimensional targets and associated instructions. Avoid generic language; aim for 'textbook quality'."

        return system_prompt + "\n\n" + user_prompt
    
    def parse(self, example_data: Dict, response: Example):
        """
        Process the LLM response into a structured format.
        
        Args:
            example_data: The original input data
            response: The LiteLLM response object
            
        Returns:
            Structured dictionary containing the generated example
        """
        dimensions = example_data['dimensions']
        topic = example_data['topic']

        generated_text = response.content if hasattr(response, 'content') else response.get('content', '')
        
        # Create the structured output
        return {
            "title": response.title if hasattr(response, 'title') else "Generated Example",
            "text": generated_text,
            "dimensions": self._serialize_dimensions(dimensions) if dimensions else {},
            "topic": topic,
            "metadata": {
                "model": self.model_name,
                "timestamp": response.created_at.isoformat() if hasattr(response, 'created_at') else None,
                "tokens": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
        }
    
    def _serialize_dimensions(self, dimensions: Dict) -> Dict[str, Any]:
        """Convert dimensions dataclass to a serializable dictionary."""
        result = {}
        for field_name, dimension_value in dimensions.items():
            choice = dimension_value['choice']
            result[field_name] = {
                "dimension": dimension_value['dimension']['name'],
                "value": choice['value'],
                "description": choice['description']
            }
        return result


def generate_test_examples(num_examples: int = MAX_EXAMPLES) -> List[Dict[str, Any]]:
    """Generate a small set of test examples with various dimension combinations."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sampler = DimensionSampler()
    generator = ContentGenerator(model_name="gemini/gemini-2.5-pro-preview-03-25")
    examples = []
    
    # Sample topics relevant to different domain emphases
    topics = [
        "Explain the process of photosynthesis and its importance to life on Earth.",
        "Discuss the historical significance of the Industrial Revolution and its impact on modern society.",
        "Describe the steps involved in creating a budget for a small household.",
        "Analyze the themes of identity and belonging in modern literature.",
        "Explore the ethical implications of artificial intelligence in healthcare decision-making."
    ]
    
    # Prepare input data for all examples
    input_examples = []
    for i in range(min(num_examples, len(topics))):
        # Sample random dimensions for this example
        dimensions = sampler.sample_example_dimensions()
        
        # Prepare the example data
        input_examples.append({
            "dimensions": dimensions.__dict__(),
            "topic": topics[i]
        })
    
    # Generate all examples
    try:
        # Use curator's batch processing capabilities with LiteLLM
        result = generator(input_examples)
        
        # Save individual examples to files
        for i, example in enumerate(result, 1):
            examples.append(example)
            with open(os.path.join(OUTPUT_DIR, f"example_{i}.json"), "w") as f:
                json.dump(example, f, indent=2)
        
        # Save all examples to a single file
        with open(os.path.join(OUTPUT_DIR, "test_examples.json"), "w") as f:
            json.dump(examples, f, indent=2)
        
    except Exception:
        Console().print_exception()
    
    return examples


def main():
    """Main function to run the test."""
    
    # Check if any required API keys are set
    api_key_env_vars = ["LITELLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", 
                         "GEMINI_API_KEY", "TOGETHER_API_KEY"]
    if not any(os.environ.get(key) for key in api_key_env_vars):
        print("\nWARNING: No API keys detected in environment variables.")
        print("Generation may fail without proper authentication.")
        sys.exit(1)
    
    examples = generate_test_examples()
    print(f"Generated {len(examples)} test examples.")
    print(f"Output saved to {OUTPUT_DIR}/test_examples.json")


if __name__ == "__main__":
    main()