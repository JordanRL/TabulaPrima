# Using `dataset_dimensions.py` for Example Generation

This document explains how the classes defined in `dataset_dimensions.py` are intended to be used within the synthetic data generation pipeline for the TabulaPrima project. These classes provide a structured way to define the characteristics of each training example, ensuring the dataset aligns with the target distributions and pedagogical goals.

## Core Concepts

The system revolves around several key dataclasses:

1.  **`DatasetDimension`**: Represents a high-level, orthogonal characteristic used to categorize data (e.g., `ContentFocusDimension`, `AmbiguityLevelDimension`). Each dimension has a `name`, a general `description`, and a list of possible `choices`.
2.  **`DimensionChoice`**: Represents a specific value within a `DatasetDimension` (e.g., `ContentFocusFactualDimensionChoice`, `AmbiguityLevelLowChoice`). Each choice includes:
    - `value`: The specific string name of the choice.
    - `description`: An explanatory description of what this choice represents.
    - `instruction`: A specific instruction tailored for a generator LLM, guiding it to produce text exhibiting this characteristic.
    - `goal`: The intended learning outcome for the model being trained on examples with this characteristic.
    - `weight`: The target fractional representation of this choice within its dimension in the final dataset.
3.  **`DatasetDimensionValue`**: A container class that links a specific `DimensionChoice` back to its parent `DatasetDimension`.
4.  **`DatasetExampleDimensions`**: A container dataclass designed to hold the selected `DatasetDimensionValue` for *each* dimension, representing the complete dimensional profile or "recipe" for a single generated training example.

## Generation Workflow Integration

Here's how these classes are typically used in a script responsible for generating training examples:

1.  **Load All Dimensions:** Start by retrieving all defined dimensions using the `get_all_dimensions()` helper function. This provides the complete framework for categorization. It's also helpful to create mappings for easier instantiation later.

    ```python
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
        # Import specific choice classes if needed directly
    )
    import random

    all_dimensions = get_all_dimensions()

    # Create mappings for convenience
    dim_name_to_instance = {dim.name: dim for dim in all_dimensions}
    dim_name_to_value_class = {
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
    dimension_value_to_field_name = {
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
    ```

2.  **Select Choices & Create Recipe Container:** For each example to be generated, sample a `DimensionChoice` for each dimension based on its weight. Then, immediately create the corresponding `DatasetDimensionValue` and populate an instance of `DatasetExampleDimensions`. This object now holds the complete "recipe" for the example.

    ```python
    dimension_values_for_example = {}

    for dim_name, dimension in dim_name_to_instance.items():
        choices = dimension.choices
        weights = [choice.weight for choice in choices]
        # Perform weighted random sampling
        selected_choice = random.choices(choices, weights=weights, k=1)[0]

        # Create the specific DimensionValue instance
        ValueClass = dim_name_to_value_class[dim_name]
        dimension_value = ValueClass(choice=selected_choice, dimension=dimension)

        # Get the field name for the DatasetExampleDimensions dataclass
        field_name = dimension_value_to_field_name[dim_name]
        dimension_values_for_example[field_name] = dimension_value

    # Instantiate the container with all selected dimension values
    example_recipe = DatasetExampleDimensions(**dimension_values_for_example)

    # 'example_recipe' now holds the complete dimensional profile for the example
    ```

3.  **Construct Generator Prompt:** Assemble a detailed prompt for the generator LLM using the information stored within the `example_recipe` object created in the previous step. Iterate through its fields to access the `instruction` and `value` from each selected choice. Combine these dimensional instructions with a specific topic, task, or seed relevant to the chosen dimensions.

    ```python
    import dataclasses

    # Example Prompt Construction (Simplified)
    prompt = "You are an expert data generator. Generate a text sample adhering strictly to the following characteristics:\n"

    # Iterate through the fields of the DatasetExampleDimensions instance
    for field_info in dataclasses.fields(example_recipe):
        field_name = field_info.name
        dimension_value: DatasetDimensionValue = getattr(example_recipe, field_name)
        choice = dimension_value.choice
        dimension = dimension_value.dimension

        prompt += f"- {dimension.name}: {choice.value}\n" # Use value for context
        prompt += f"  Instruction: {choice.instruction}\n" # Key instruction from the choice

    # Add a specific topic/task instruction
    # (This topic itself might be sampled based on the chosen DomainEmphasis, etc.)
    prompt += "\nTopic/Task: Explain the concept of photosynthesis for a high school biology class, ensuring high factual density and providing detailed step-by-step reasoning (CoT)."

    # 'prompt' is now ready to be sent to the generator LLM.
    ```

4.  **Generate Example:** Send the constructed `prompt` to the chosen generator LLM API and receive the generated text.

5.  **Store Metadata:** The `example_recipe` object created in Step 2 already contains all the necessary dimensional metadata. Store this object directly alongside the generated text. This structured metadata is vital for:
    - **Dataset Analysis:** Understanding the composition of the generated data across all dimensions.
    - **Filtering/Balancing:** Ensuring the actual generated dataset matches the target distributions.
    - **Curriculum Learning:** Selecting or ordering examples based on specific dimensions (like `conceptual_complexity` or `reasoning`) during training.

    ```python
    # Store 'generated_text' and the 'example_recipe' object together
    # (e.g., in a dictionary, database record, or Hugging Face Dataset row)
    final_example = {
        "text": generated_text, # Assume this variable holds the LLM output
        "dimensions": example_recipe # The object created in Step 2
    }
    ```

By following this workflow, the `dataset_dimensions.py` classes, especially the `DatasetExampleDimensions` container, provide a clean and efficient way to define, manage, and utilize the dimensional characteristics for each generated example. This systematic approach is key to creating the targeted high-density dataset and enabling the planned training and curriculum learning strategies.