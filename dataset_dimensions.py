# TabulaPrima/dataset_dimensions.py

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass(frozen=True)
class DimensionChoice:
    """Represents a specific value within a dataset dimension."""
    value: str
    description: str # Explanatory description of the data characteristic for this choice.
    instruction: str # Instructive text for a generator model creating examples with this choice.
    goal: str        # Intended learning outcome for the model from examples with this choice.
    weight: float    # Target proportion of the dataset having this choice for the dimension.

@dataclass(frozen=True)
class DatasetDimension:
    """Represents an orthogonal dimension used to categorize dataset examples."""
    name: str
    description: str # Explanation of the aspect captured by this dimension.
    choices: list[DimensionChoice]

@dataclass
class DatasetDimensionValue:
    """Associates a specific choice with its parent dimension."""
    choice: DimensionChoice
    dimension: DatasetDimension

# -----------------------------
# Dimension 1: Content Focus
# -----------------------------

@dataclass(frozen=True)
class ContentFocusDimensionChoice(DimensionChoice):
    """Base class for Content Focus choices."""
    value: Literal[
        "Factual Knowledge & Explanation",
        "Procedural & Task-Oriented",
        "Reasoning & Logic Exposition",
        "Argumentation & Multi-Perspective Analysis",
        "Creative & Narrative Expression",
        "Meta-Cognitive & Self-Correction"
    ]

@dataclass(frozen=True)
class ContentFocusFactualDimensionChoice(ContentFocusDimensionChoice):
    value: Literal["Factual Knowledge & Explanation"] = "Factual Knowledge & Explanation"
    description: str = "The text primarily conveys factual information, explains concepts, or describes processes objectively."
    instruction: str = "Generate textbook-style explanations, definitions of concepts, or descriptions of processes (e.g., science, history, technical fields)."
    goal: str = "To build the model's foundational world knowledge and ability to explain concepts clearly."
    weight: float = 0.3

@dataclass(frozen=True)
class ContentFocusProceduralDimensionChoice(ContentFocusDimensionChoice):
    value: Literal["Procedural & Task-Oriented"] = "Procedural & Task-Oriented"
    description: str = "The text provides step-by-step guidance, instructions for completing a task, or demonstrates a process."
    instruction: str = "Generate step-by-step instructions, detailed walkthroughs for solving problems, coding examples, or explanations of function usage."
    goal: str = "To develop the model's ability to follow instructions accurately and execute defined tasks."
    weight: float = 0.2

@dataclass(frozen=True)
class ContentFocusReasoningDimensionChoice(ContentFocusDimensionChoice):
    value: Literal["Reasoning & Logic Exposition"] = "Reasoning & Logic Exposition"
    description: str = "The text explicitly demonstrates a logical argument, proof, or chain of reasoning (e.g., deduction, induction)."
    instruction: str = "Generate explicit demonstrations of reasoning chains (deductive, inductive, causal, probabilistic), mathematical proofs, or solutions to logical puzzles."
    goal: str = "To explicitly teach the model structured thinking and logical deduction/inference patterns."
    weight: float = 0.2

@dataclass(frozen=True)
class ContentFocusArgumentationDimensionChoice(ContentFocusDimensionChoice):
    value: Literal["Argumentation & Multi-Perspective Analysis"] = "Argumentation & Multi-Perspective Analysis"
    description: str = "The text presents and analyzes different viewpoints, arguments, or perspectives on a specific topic."
    instruction: str = "Generate debates, comparative analyses, or explorations of different viewpoints on a topic, potentially including ethical or controversial issues."
    goal: str = "To develop the model's ability to understand nuance, evaluate arguments, and analyze topics from multiple perspectives."
    weight: float = 0.15

@dataclass(frozen=True)
class ContentFocusCreativeDimensionChoice(ContentFocusDimensionChoice):
    value: Literal["Creative & Narrative Expression"] = "Creative & Narrative Expression"
    description: str = "The text focuses on storytelling, creative writing, dialogue, poetry, or expressing subjective experiences."
    instruction: str = "Generate stories, dialogues, poetry, or reflective passages focusing on subjective experience or imagination."
    goal: str = "To enhance the model's stylistic flexibility, fluency, and understanding of human context and narrative."
    weight: float = 0.1

@dataclass(frozen=True)
class ContentFocusMetaCognitiveDimensionChoice(ContentFocusDimensionChoice):
    value: Literal["Meta-Cognitive & Self-Correction"] = "Meta-Cognitive & Self-Correction"
    description: str = "The text includes reflections on the thinking process itself, acknowledges limitations, corrects errors, or expresses uncertainty."
    instruction: str = "Generate text that reflects on its own thought process, identifies potential errors, refines arguments, or expresses uncertainty."
    goal: str = "To foster model self-awareness, the ability to self-critique, and express degrees of confidence or uncertainty."
    weight: float = 0.05

@dataclass(frozen=True)
class ContentFocusDimension(DatasetDimension):
    name: str = "Content Focus"
    description: str = "Categorizes the primary type of information or implicit communicative goal of the text."
    choices: list[ContentFocusDimensionChoice] = field(
        default_factory=lambda: [
            ContentFocusFactualDimensionChoice(),
            ContentFocusProceduralDimensionChoice(),
            ContentFocusReasoningDimensionChoice(),
            ContentFocusArgumentationDimensionChoice(),
            ContentFocusCreativeDimensionChoice(),
            ContentFocusMetaCognitiveDimensionChoice()
        ]
    )

@dataclass
class ContentFocusDimensionValue(DatasetDimensionValue):
    choice: Union[
        ContentFocusFactualDimensionChoice,
        ContentFocusProceduralDimensionChoice,
        ContentFocusReasoningDimensionChoice,
        ContentFocusArgumentationDimensionChoice,
        ContentFocusCreativeDimensionChoice,
        ContentFocusMetaCognitiveDimensionChoice
    ]
    dimension: ContentFocusDimension = field(default_factory=lambda: ContentFocusDimension())

# -----------------------------
# Dimension 2: Format & Structure
# -----------------------------

@dataclass(frozen=True)
class FormatDimensionChoice(DimensionChoice):
    """Base class for Format & Structure choices."""
    value: Literal[
        "Structured Prose",
        "Dialogue/Transcript",
        "Instructional/Step-by-Step",
        "Question-Answer (Direct)",
        "Question-Answer (with Rationale/CoT)",
        "Fragmented/Note-like"
    ]

@dataclass(frozen=True)
class FormatStructuredProseDimensionChoice(FormatDimensionChoice):
    value: Literal["Structured Prose"] = "Structured Prose"
    description: str = "The text is organized as a standard written document like an essay, article, or report with coherent paragraphs and sections."
    instruction: str = "Generate text formatted as essays, articles, or reports with clear organizational patterns (e.g., introduction, body, conclusion)."
    goal: str = "To teach the model to understand and generate well-structured, longer-form text for conveying complex information."
    weight: float = 0.3

@dataclass(frozen=True)
class FormatDialogueTranscriptDimensionChoice(FormatDimensionChoice):
    value: Literal["Dialogue/Transcript"] = "Dialogue/Transcript"
    description: str = "The text represents a conversation or spoken exchange between two or more participants."
    instruction: str = "Generate text representing multi-turn conversational exchanges between speakers."
    goal: str = "To develop the model's conversational abilities and understanding of interaction dynamics."
    weight: float = 0.15

@dataclass(frozen=True)
class FormatInstructionalDimensionChoice(FormatDimensionChoice):
    value: Literal["Instructional/Step-by-Step"] = "Instructional/Step-by-Step"
    description: str = "The text is formatted as a sequence of steps, commands, or instructions, often using lists or code blocks."
    instruction: str = "Generate text formatted as numbered lists, code blocks with explanations, or sequential guides."
    goal: str = "To improve the model's ability to follow and generate clear, procedural instructions."
    weight: float = 0.15

@dataclass(frozen=True)
class FormatQuestionAnswerDirectDimensionChoice(FormatDimensionChoice):
    value: Literal["Question-Answer (Direct)"] = "Question-Answer (Direct)"
    description: str = "The text consists of distinct questions immediately followed by concise answers, without elaboration."
    instruction: str = "Generate focused question-and-answer pairs without providing extended reasoning or context."
    goal: str = "To teach the model direct factual recall and targeted information retrieval in a Q&A format."
    weight: float = 0.1

@dataclass(frozen=True)
class FormatQuestionAnswerCotDimensionChoice(FormatDimensionChoice):
    value: Literal["Question-Answer (with Rationale/CoT)"] = "Question-Answer (with Rationale/CoT)"
    description: str = "The text presents questions followed by answers that include an explicit step-by-step explanation of the reasoning process (Chain-of-Thought)."
    instruction: str = "Generate question-and-answer pairs where the answer explicitly includes the step-by-step reasoning (Chain-of-Thought) used to arrive at it."
    goal: str = "To explicitly teach the model how to show its reasoning process when answering questions."
    weight: float = 0.25

@dataclass(frozen=True)
class FormatFragmentedNoteDimensionChoice(FormatDimensionChoice):
    value: Literal["Fragmented/Note-like"] = "Fragmented/Note-like"
    description: str = "The text uses an informal structure, possibly with bullet points, incomplete sentences, or abbreviations, resembling notes or draft ideas."
    instruction: str = "Generate less formal text, potentially using bullet points, incomplete sentences, or abbreviations, simulating note-taking or draft thoughts."
    goal: str = "To expose the model to less formal structures and simulate internal thought processes, adding variety (use sparingly)."
    weight: float = 0.05

@dataclass(frozen=True)
class FormatDimension(DatasetDimension):
    name: str = "Format & Structure"
    description: str = "Defines the organizational layout and presentation style of the text example."
    choices: list[FormatDimensionChoice] = field(
        default_factory=lambda: [
            FormatStructuredProseDimensionChoice(),
            FormatDialogueTranscriptDimensionChoice(),
            FormatInstructionalDimensionChoice(),
            FormatQuestionAnswerDirectDimensionChoice(),
            FormatQuestionAnswerCotDimensionChoice(),
            FormatFragmentedNoteDimensionChoice()
        ]
    )

@dataclass
class FormatDimensionValue(DatasetDimensionValue):
    choice: Union[
        FormatStructuredProseDimensionChoice,
        FormatDialogueTranscriptDimensionChoice,
        FormatInstructionalDimensionChoice,
        FormatQuestionAnswerDirectDimensionChoice,
        FormatQuestionAnswerCotDimensionChoice,
        FormatFragmentedNoteDimensionChoice
    ]
    dimension: FormatDimension = field(default_factory=lambda: FormatDimension())

# -----------------------------
# Dimension 3: Domain Emphasis
# -----------------------------

@dataclass(frozen=True)
class DomainEmphasisDimensionChoice(DimensionChoice):
    """Base class for Domain Emphasis choices."""
    value: Literal[
        "STEM (Science, Technology, Engineering, Math)",
        "Humanities & Social Sciences",
        "Practical & Applied",
        "Creative Arts & Literature",
        "Abstract Reasoning & Philosophy"
    ]

@dataclass(frozen=True)
class DomainEmphasisStemDimensionChoice(DomainEmphasisDimensionChoice):
    value: Literal["STEM (Science, Technology, Engineering, Math)"] = "STEM (Science, Technology, Engineering, Math)"
    description: str = "The text deals with topics in science, technology, engineering, or mathematics, often involving logical, quantitative, or causal reasoning."
    instruction: str = "Generate text focusing on logical, quantitative, and causal reasoning within scientific, technological, engineering, or mathematical fields."
    goal: str = "To build strong logical/quantitative skills and structured knowledge of technical concepts."
    weight: float = 0.35

@dataclass(frozen=True)
class DomainEmphasisHumanitiesDimensionChoice(DomainEmphasisDimensionChoice):
    value: Literal["Humanities & Social Sciences"] = "Humanities & Social Sciences"
    description: str = "The text covers topics in the humanities (e.g., literature, history, arts) or social sciences (e.g., sociology, psychology), emphasizing interpretation and context."
    instruction: str = "Generate text emphasizing interpretation, argumentation, historical analysis, or cultural context within humanities or social science domains."
    goal: str = "To build understanding of human behavior, social interactions, cultural/historical context, and ethical considerations."
    weight: float = 0.25

@dataclass(frozen=True)
class DomainEmphasisPracticalDimensionChoice(DomainEmphasisDimensionChoice):
    value: Literal["Practical & Applied"] = "Practical & Applied"
    description: str = "The text relates to everyday activities, common knowledge, practical skills, how-to instructions, or safety."
    instruction: str = "Generate text covering everyday knowledge, practical skills, how-to guides, safety procedures, or common sense reasoning."
    goal: str = "To ground the model's knowledge in real-world tasks and common sense applications."
    weight: float = 0.15

@dataclass(frozen=True)
class DomainEmphasisCreativeDimensionChoice(DomainEmphasisDimensionChoice):
    value: Literal["Creative Arts & Literature"] = "Creative Arts & Literature"
    description: str = "The text involves creative writing, literary analysis, discussion of arts, or expressions of emotion and imagination."
    instruction: str = "Generate text focusing on creative writing (stories, poems), artistic representation, literary analysis, or emotional expression."
    goal: str = "To enhance stylistic flexibility, narrative understanding, and comprehension of diverse human expression."
    weight: float = 0.1

@dataclass(frozen=True)
class DomainEmphasisAbstractDimensionChoice(DomainEmphasisDimensionChoice):
    value: Literal["Abstract Reasoning & Philosophy"] = "Abstract Reasoning & Philosophy"
    description: str = "The text explores abstract concepts, logical reasoning, philosophical arguments, ethical dilemmas, or meta-cognitive ideas."
    instruction: str = "Generate text dealing with abstract logic, ethical dilemmas, meta-cognitive reflection, philosophical concepts, or fundamental principles."
    goal: str = "To foster advanced capabilities in meta-cognition, ethical reasoning, and complex logical thought."
    weight: float = 0.15

@dataclass(frozen=True)
class DomainEmphasisDimension(DatasetDimension):
    name: str = "Domain Emphasis"
    description: str = "Describes the primary subject matter or knowledge area the text example focuses on."
    choices: list[DomainEmphasisDimensionChoice] = field(
        default_factory=lambda: [
            DomainEmphasisStemDimensionChoice(),
            DomainEmphasisHumanitiesDimensionChoice(),
            DomainEmphasisPracticalDimensionChoice(),
            DomainEmphasisCreativeDimensionChoice(),
            DomainEmphasisAbstractDimensionChoice()
        ]
    )

@dataclass
class DomainEmphasisDimensionValue(DatasetDimensionValue):
    choice: Union[
        DomainEmphasisStemDimensionChoice,
        DomainEmphasisHumanitiesDimensionChoice,
        DomainEmphasisPracticalDimensionChoice,
        DomainEmphasisCreativeDimensionChoice,
        DomainEmphasisAbstractDimensionChoice,
    ]
    dimension: DomainEmphasisDimension = field(default_factory=lambda: DomainEmphasisDimension())


# -----------------------------
# Dimension 4: Reasoning Explicitness
# -----------------------------

@dataclass(frozen=True)
class ReasoningDimensionChoice(DimensionChoice):
    """Base class for Reasoning Explicitness choices."""
    value: Literal[
        "Implicit Reasoning",
        "Concise Reasoning",
        "Detailed Chain-of-Thought (CoT)",
        "Self-Reflective/Corrective Reasoning",
        "Probabilistic Reasoning" # Added back per user request
    ]

@dataclass(frozen=True)
class ReasoningImplicitDimensionChoice(ReasoningDimensionChoice):
    value: Literal["Implicit Reasoning"] = "Implicit Reasoning"
    description: str = "The reasoning process behind conclusions or answers is not explicitly stated in the text."
    instruction: str = "Generate text where the conclusion/answer is provided with minimal or no explicit justification (typical of standard web text)."
    goal: str = "To provide a baseline for fluency on standard text, but minimize exposure compared to web data."
    weight: float = 0.2

@dataclass(frozen=True)
class ReasoningConciseDimensionChoice(ReasoningDimensionChoice):
    value: Literal["Concise Reasoning"] = "Concise Reasoning"
    description: str = "The text provides a brief summary or key steps of the reasoning process, but not a full detailed breakdown."
    instruction: str = "Generate text giving a few key steps or justifications, balancing implicitness and full detail."
    goal: str = "To teach common patterns in well-written explanations and bridge towards detailed reasoning."
    weight: float = 0.25

@dataclass(frozen=True)
class ReasoningCoTDimensionChoice(ReasoningDimensionChoice):
    value: Literal["Detailed Chain-of-Thought (CoT)"] = "Detailed Chain-of-Thought (CoT)"
    description: str = "The text includes a detailed, explicit, step-by-step explanation of the thinking process used to reach the conclusion."
    instruction: str = "Generate text providing an explicit, step-by-step breakdown of the thinking process from premise to conclusion."
    goal: str = "To force the model to learn explicit reasoning patterns; a core focus."
    weight: float = 0.3

@dataclass(frozen=True)
class ReasoningSelfReflectiveDimensionChoice(ReasoningDimensionChoice):
    value: Literal["Self-Reflective/Corrective Reasoning"] = "Self-Reflective/Corrective Reasoning"
    description: str = "The text not only shows reasoning steps but also includes meta-commentary on the process, like identifying uncertainties, potential flaws, or corrections."
    instruction: str = "Generate Chain-of-Thought text that also includes identifying flaws, expressing uncertainty, considering alternatives, or refining the process."
    goal: str = "To promote model robustness, self-awareness, uncertainty estimation, and corrigibility."
    weight: float = 0.1

@dataclass(frozen=True)
class ReasoningProbabilisticDimensionChoice(ReasoningDimensionChoice):
    value: Literal["Probabilistic Reasoning"] = "Probabilistic Reasoning"
    description: str = "The text involves reasoning based on likelihoods, probabilities, statistical evidence, or dealing with uncertainty quantitatively."
    instruction: str = "Generate text that reasons about likelihoods, probabilities, or statistical evidence when dealing with uncertainty."
    goal: str = "To teach the model how to reason with uncertainty or incomplete information using probabilistic concepts."
    weight: float = 0.15

@dataclass(frozen=True)
class ReasoningDimension(DatasetDimension):
    name: str = "Reasoning Explicitness"
    description: str = "Measures the degree to which the underlying thinking process is overtly detailed in the text."
    choices: list[ReasoningDimensionChoice] = field(
        default_factory=lambda: [
            ReasoningImplicitDimensionChoice(),
            ReasoningConciseDimensionChoice(),
            ReasoningCoTDimensionChoice(),
            ReasoningSelfReflectiveDimensionChoice(),
            ReasoningProbabilisticDimensionChoice() # Added back
        ]
    )

@dataclass
class ReasoningDimensionValue(DatasetDimensionValue):
    choice: Union[
        ReasoningImplicitDimensionChoice,
        ReasoningConciseDimensionChoice,
        ReasoningCoTDimensionChoice,
        ReasoningSelfReflectiveDimensionChoice,
        ReasoningProbabilisticDimensionChoice, # Added back
    ]
    dimension: ReasoningDimension = field(default_factory=lambda: ReasoningDimension())


# ----------------------------------
# Dimension 5: Conceptual Complexity
# ----------------------------------

@dataclass(frozen=True)
class ConceptualComplexityDimensionChoice(DimensionChoice):
    """Base class for Conceptual Complexity choices."""
    value: Literal[
        "Basic Concepts",
        "Intermediate Concepts",
        "Advanced/Interdisciplinary Concepts"
    ]

@dataclass(frozen=True)
class ConceptualComplexityBasicChoice(ConceptualComplexityDimensionChoice):
    value: Literal["Basic Concepts"] = "Basic Concepts"
    description: str = "The text deals with simple, self-contained ideas that are easy to understand without requiring significant prior knowledge."
    instruction: str = "Generate text explaining simple, self-contained concepts or topics requiring minimal prerequisite knowledge."
    goal: str = "To build the model's foundational knowledge base, necessary for understanding more complex ideas."
    weight: float = 0.3

@dataclass(frozen=True)
class ConceptualComplexityIntermediateChoice(ConceptualComplexityDimensionChoice):
    value: Literal["Intermediate Concepts"] = "Intermediate Concepts"
    description: str = "The text discusses topics that require connecting multiple basic ideas or assume some background knowledge."
    instruction: str = "Generate text on topics that require linking several basic ideas or assume some prerequisite knowledge."
    goal: str = "To teach the model synthesis and connection-making between concepts; forms the bulk of learning."
    weight: float = 0.4

@dataclass(frozen=True)
class ConceptualComplexityAdvancedChoice(ConceptualComplexityDimensionChoice):
    value: Literal["Advanced/Interdisciplinary Concepts"] = "Advanced/Interdisciplinary Concepts"
    description: str = "The text covers highly complex, abstract, or specialized topics, potentially integrating knowledge from multiple fields."
    instruction: str = "Generate text that integrates knowledge from multiple domains or involves highly abstract or complex ideas."
    goal: str = "To push the model towards deeper understanding, integrating diverse knowledge, and handling high complexity."
    weight: float = 0.3

@dataclass(frozen=True)
class ConceptualComplexityDimension(DatasetDimension):
    name: str = "Conceptual Complexity"
    description: str = "Assesses the difficulty and interconnectedness of the ideas presented in the text."
    choices: list[ConceptualComplexityDimensionChoice] = field(
        default_factory=lambda: [
            ConceptualComplexityBasicChoice(),
            ConceptualComplexityIntermediateChoice(),
            ConceptualComplexityAdvancedChoice()
        ]
    )

@dataclass
class ConceptualComplexityDimensionValue(DatasetDimensionValue):
    choice: Union[
        ConceptualComplexityBasicChoice,
        ConceptualComplexityIntermediateChoice,
        ConceptualComplexityAdvancedChoice
    ]
    dimension: ConceptualComplexityDimension = field(default_factory=lambda: ConceptualComplexityDimension())

# -----------------------------
# Dimension 6: Ambiguity Level
# -----------------------------

@dataclass(frozen=True)
class AmbiguityLevelDimensionChoice(DimensionChoice):
    """Base class for Ambiguity Level choices."""
    value: Literal[
        "Low Ambiguity",
        "Moderate Ambiguity",
        "High Ambiguity"
    ]

@dataclass(frozen=True)
class AmbiguityLevelLowChoice(AmbiguityLevelDimensionChoice):
    value: Literal["Low Ambiguity"] = "Low Ambiguity"
    description: str = "The text describes situations, questions, or instructions with clear, single interpretations and definite answers/outcomes."
    instruction: str = "Generate text describing clear-cut situations, factual questions with single answers, or unambiguous instructions."
    goal: str = "To ground the model in handling facts, clear procedures, and situations with definite outcomes."
    weight: float = 0.4

@dataclass(frozen=True)
class AmbiguityLevelModerateChoice(AmbiguityLevelDimensionChoice):
    value: Literal["Moderate Ambiguity"] = "Moderate Ambiguity"
    description: str = "The text presents scenarios requiring some interpretation, consideration of nuance, or weighing limited factors."
    instruction: str = "Generate text involving situations that require some interpretation, nuance, or weighing limited factors."
    goal: str = "To develop the model's nuance, interpretation skills, and ability to handle underspecified situations."
    weight: float = 0.35

@dataclass(frozen=True)
class AmbiguityLevelHighChoice(AmbiguityLevelDimensionChoice):
    value: Literal["High Ambiguity"] = "High Ambiguity"
    description: str = "The text deals with situations having multiple valid interpretations, inherent uncertainty, complex trade-offs, or subjective elements (e.g., ethics, art)."
    instruction: str = "Generate text presenting scenarios with multiple valid interpretations, inherent uncertainty, complex trade-offs (e.g., ethical dilemmas, creative tasks)."
    goal: str = "To teach the model to handle complex decision-making, ethical reasoning, and situations without single 'correct' answers."
    weight: float = 0.25

@dataclass(frozen=True)
class AmbiguityLevelDimension(DatasetDimension):
    name: str = "Ambiguity Level"
    description: str = "Quantifies the degree of uncertainty or potential for multiple interpretations in the text."
    choices: list[AmbiguityLevelDimensionChoice] = field(
        default_factory=lambda: [
            AmbiguityLevelLowChoice(),
            AmbiguityLevelModerateChoice(),
            AmbiguityLevelHighChoice()
        ]
    )

@dataclass
class AmbiguityLevelDimensionValue(DatasetDimensionValue):
    choice: Union[
        AmbiguityLevelLowChoice,
        AmbiguityLevelModerateChoice,
        AmbiguityLevelHighChoice
    ]
    dimension: AmbiguityLevelDimension = field(default_factory=lambda: AmbiguityLevelDimension())

# -----------------------------
# Dimension 7: Factual Density
# -----------------------------

@dataclass(frozen=True)
class FactualDensityDimensionChoice(DimensionChoice):
    """Base class for Factual Density choices."""
    value: Literal[
        "Low Density",
        "Moderate Density",
        "High Density"
    ]

@dataclass(frozen=True)
class FactualDensityLowChoice(FactualDensityDimensionChoice):
    value: Literal["Low Density"] = "Low Density"
    description: str = "The text contains few objectively verifiable facts, focusing more on opinions, narrative, or abstract ideas."
    instruction: str = "Generate text that is abstract, opinion-based, narrative-focused, or contains few verifiable facts."
    goal: str = "To ensure the model can handle narrative, opinion, and abstract reasoning where specific facts are secondary."
    weight: float = 0.2

@dataclass(frozen=True)
class FactualDensityModerateChoice(FactualDensityDimensionChoice):
    value: Literal["Moderate Density"] = "Moderate Density"
    description: str = "The text includes a balanced mix of factual statements and narrative, opinion, or explanatory content."
    instruction: str = "Generate text containing a mix of narrative/opinion and verifiable facts, data, or evidence."
    goal: str = "To teach the model to handle balanced explanations and arguments supported by some evidence."
    weight: float = 0.4

@dataclass(frozen=True)
class FactualDensityHighChoice(FactualDensityDimensionChoice):
    value: Literal["High Density"] = "High Density"
    description: str = "The text is primarily composed of verifiable facts, data points, evidence, or specific citations."
    instruction: str = "Generate text primarily focused on presenting verifiable facts, data, evidence, or citations, with minimal opinion or narrative."
    goal: str = "To emphasize grounding, factual recall, and evidence-based reasoning."
    weight: float = 0.4

@dataclass(frozen=True)
class FactualDensityDimension(DatasetDimension):
    name: str = "Factual Density"
    description: str = "Measures the concentration of verifiable facts, data, or evidence within the text."
    choices: list[FactualDensityDimensionChoice] = field(
        default_factory=lambda: [
            FactualDensityLowChoice(),
            FactualDensityModerateChoice(),
            FactualDensityHighChoice()
        ]
    )

@dataclass
class FactualDensityDimensionValue(DatasetDimensionValue):
    choice: Union[
        FactualDensityLowChoice,
        FactualDensityModerateChoice,
        FactualDensityHighChoice
    ]
    dimension: FactualDensityDimension = field(default_factory=lambda: FactualDensityDimension())

# -----------------------------
# Dimension 8: Temporal Focus
# -----------------------------

@dataclass(frozen=True)
class TemporalFocusDimensionChoice(DimensionChoice):
    """Base class for Temporal Focus choices."""
    value: Literal[
        "Timeless/Foundational",
        "Contemporary/Dynamic"
    ]

@dataclass(frozen=True)
class TemporalFocusTimelessChoice(TemporalFocusDimensionChoice):
    value: Literal["Timeless/Foundational"] = "Timeless/Foundational"
    description: str = "The content discusses established principles, historical events, or core knowledge not subject to frequent change."
    instruction: str = "Generate text covering core principles, historical events, established knowledge, or concepts not expected to change quickly."
    goal: str = "To focus model learning on enduring knowledge and reasoning principles, reducing susceptibility to data staleness."
    weight: float = 0.7

@dataclass(frozen=True)
class TemporalFocusContemporaryChoice(TemporalFocusDimensionChoice):
    value: Literal["Contemporary/Dynamic"] = "Contemporary/Dynamic"
    description: str = "The content relates to current events, recent developments, or information that is time-sensitive and may change."
    instruction: str = "Generate text discussing current events, recent research, evolving situations, or topics requiring awareness of recency."
    goal: str = "To ensure the model has awareness of current context while avoiding overfitting to transient information."
    weight: float = 0.3

@dataclass(frozen=True)
class TemporalFocusDimension(DatasetDimension):
    name: str = "Temporal Focus"
    description: str = "Indicates whether the content pertains to timeless principles or time-sensitive information."
    choices: list[TemporalFocusDimensionChoice] = field(
        default_factory=lambda: [
            TemporalFocusTimelessChoice(),
            TemporalFocusContemporaryChoice()
        ]
    )

@dataclass
class TemporalFocusDimensionValue(DatasetDimensionValue):
    choice: Union[
        TemporalFocusTimelessChoice,
        TemporalFocusContemporaryChoice
    ]
    dimension: TemporalFocusDimension = field(default_factory=lambda: TemporalFocusDimension())

# --------------------------------------
# Dimension 9: Emotional Valence & Style
# --------------------------------------

@dataclass(frozen=True)
class EmotionalValenceStyleDimensionChoice(DimensionChoice):
    """Base class for Emotional Valence & Style choices."""
    value: Literal[
        "Neutral/Objective",
        "Positive/Constructive",
        "Negative/Critical",
        "Empathetic/Personal",
        "Creative/Figurative"
    ]

@dataclass(frozen=True)
class EmotionalValenceStyleNeutralChoice(EmotionalValenceStyleDimensionChoice):
    value: Literal["Neutral/Objective"] = "Neutral/Objective"
    description: str = "The text maintains a formal, impartial, and informational tone, avoiding emotional language or subjective bias."
    instruction: str = "Generate text with a formal, detached, informational tone, avoiding emotional language or subjective opinion."
    goal: str = "To establish the dominant style for factual, technical, and reasoning-heavy content."
    weight: float = 0.5

@dataclass(frozen=True)
class EmotionalValenceStylePositiveChoice(EmotionalValenceStyleDimensionChoice):
    value: Literal["Positive/Constructive"] = "Positive/Constructive"
    description: str = "The text conveys an encouraging, supportive, optimistic, or solution-focused sentiment."
    instruction: str = "Generate text with an encouraging, supportive, optimistic, or solution-oriented tone."
    goal: str = "To enable the model to generate helpful, motivating, and constructive responses."
    weight: float = 0.15

@dataclass(frozen=True)
class EmotionalValenceStyleNegativeChoice(EmotionalValenceStyleDimensionChoice):
    value: Literal["Negative/Critical"] = "Negative/Critical"
    description: str = "The text expresses skepticism, caution, criticism, disagreement, or a negative evaluation."
    instruction: str = "Generate text expressing a questioning, skeptical, cautionary, critical, or dissenting tone."
    goal: str = "To develop the model's skills in analysis, identifying flaws, and expressing caution appropriately."
    weight: float = 0.1

@dataclass(frozen=True)
class EmotionalValenceStyleEmpatheticChoice(EmotionalValenceStyleDimensionChoice):
    value: Literal["Empathetic/Personal"] = "Empathetic/Personal"
    description: str = "The text shows understanding of or shares subjective feelings, perspectives, or personal experiences."
    instruction: str = "Generate text expressing understanding, reflecting on subjective experience, or adopting a personal perspective."
    goal: str = "To develop the model's understanding of human perspectives, emotional context, and reflective dialogue."
    weight: float = 0.15

@dataclass(frozen=True)
class EmotionalValenceStyleCreativeChoice(EmotionalValenceStyleDimensionChoice):
    value: Literal["Creative/Figurative"] = "Creative/Figurative"
    description: str = "The text uses literary techniques, figurative language (metaphors, similes), humor, or unconventional stylistic elements."
    instruction: str = "Generate text employing literary devices, metaphors, analogies, humor, or other non-literal and varied styles."
    goal: str = "To enhance the model's stylistic range and its understanding of figurative and non-literal language."
    weight: float = 0.1

@dataclass(frozen=True)
class EmotionalValenceStyleDimension(DatasetDimension):
    name: str = "Emotional Valence & Style"
    description: str = "Captures the overall tone, sentiment, and stylistic nature of the text's expression."
    choices: list[EmotionalValenceStyleDimensionChoice] = field(
        default_factory=lambda: [
            EmotionalValenceStyleNeutralChoice(),
            EmotionalValenceStylePositiveChoice(),
            EmotionalValenceStyleNegativeChoice(),
            EmotionalValenceStyleEmpatheticChoice(),
            EmotionalValenceStyleCreativeChoice()
        ]
    )

@dataclass
class EmotionalValenceStyleDimensionValue(DatasetDimensionValue):
    choice: Union[
        EmotionalValenceStyleNeutralChoice,
        EmotionalValenceStylePositiveChoice,
        EmotionalValenceStyleNegativeChoice,
        EmotionalValenceStyleEmpatheticChoice,
        EmotionalValenceStyleCreativeChoice
    ]
    dimension: EmotionalValenceStyleDimension = field(default_factory=lambda: EmotionalValenceStyleDimension())

# ------------------------------------
# Dataset Example Dimensions container
# ------------------------------------

@dataclass
class DatasetExampleDimensions:
    content_focus: ContentFocusDimensionValue
    reasoning: ReasoningDimensionValue
    conceptual_complexity: ConceptualComplexityDimensionValue
    ambiguity_level: AmbiguityLevelDimensionValue
    factual_density: FactualDensityDimensionValue
    temporal_focus: TemporalFocusDimensionValue
    emotional_valence_style: EmotionalValenceStyleDimensionValue

# --- Helper function to get all dimensions ---
def get_all_dimensions() -> list[DatasetDimension]:
    """Returns a list of all defined dataset dimensions."""
    return [
        ContentFocusDimension(),
        FormatDimension(),
        DomainEmphasisDimension(),
        ReasoningDimension(),
        ConceptualComplexityDimension(),
        AmbiguityLevelDimension(),
        FactualDensityDimension(),
        TemporalFocusDimension(),
        EmotionalValenceStyleDimension()
    ]