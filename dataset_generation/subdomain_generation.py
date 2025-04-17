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

class TopicVariations(BaseModel):
    variation: str = Field(description="The variation on the topic.")
    variation_type: str = Field(description="How the topic was varied. E.g., By focusing on the emotional elements or By looking at causes.")

class Topic(BaseModel):
    domain: str = Field(description="The domain of the topic.")
    sub_domain: str = Field(description="The sub-domain of the topic.")
    topic: str = Field(description="The topic name.")
    description: str = Field(description="A brief description of the topic.")
    variations: List[TopicVariations] = Field(description="A list of variations on the topic, applicable to different contexts or learning scenarios.")

class Subdomains(BaseModel):
    subdomains: List[str] = Field(description="A list of subdomains.")

class SubdomainGenerator(curator.LLM):
    response_format = Subdomains

    def prompt(self, domain: Dict):
        prompt = f"You are an expert data generator tasked with creating high-quality pre-training examples for a large language model."
        prompt += f" Your goal is to help ensure broad coverage within training data."
        prompt += f" The following domain is being considered:\n\n"
        prompt += f"Domain: {domain['domain']}\n"
        prompt += f"Description: {domain['description']}\n"
        prompt += f"Training Goal: {domain['goal']}\n\n"
        prompt += f"Create a list of subdomains that are relevant to the domain. Each subdomain should be a single word or phrase."
        return prompt

    def parse(self, domain: Dict, response: Subdomains) -> List[Dict[str, Any]]:
        return [{"subdomain": s, "domain": domain['domain'], "description": domain['description'], "goal": domain['goal']} for s in response.subdomains]

class TopicGenerator(curator.LLM):
    response_format = Topic

    def prompt(self, topic_info: Dict):
        prompt = f"You are an expert data generator tasked with creating high-quality pre-training examples for a large language model."
        prompt += f" Your goal is to help ensure broad coverage within training data."
        prompt += f" The following domain is being considered:\n\n"
        prompt += f"Domain: {topic_info['domain']}\n"
        prompt += f"- Description: {topic_info['description']}\n"
        prompt += f"- Training Goal: {topic_info['goal']}\n"
        prompt += f"- Sub-Domain: {topic_info['subdomain']}\n\n"
        prompt += f"Create a list of topics that are relevant to the sub-domain. Each topic should be specific, but topics do not need to be elements of knowledge"
        prompt += f" They must simply be relevant to the domain and sub-domain.\n\n"
        prompt += f"For each topic, provide several variations that are relevant to different contexts or learning scenarios."
        prompt += f" A variation is not a different topic, but rather a different way of considering the same topic."
        return prompt

    def parse(self, topic_info: Dict, response: Topic) -> Dict[str, Any]:
        return response.model_dump(mode="json")

def main():
    """Main function to run the test."""
    domain_dimension = DomainEmphasisDimension()
    domains = domain_dimension.choices

    subdomain_generator = SubdomainGenerator(
        model_name="gemini/gemini-2.0-flash",
        backend="litellm",
        backend_params={
            "max_requests_per_minute": 2_000,  # Rate limit for requests
            "max_tokens_per_minute": 4_000_000  # Token usage limit
        }
    )
    topic_generator = TopicGenerator(
        model_name="gpt-4o-mini"
    )

    domain_info = []
    topic_info = []
    for domain in domains:
        domain_info.append({
            "domain": domain.value,
            "description": domain.description,
            "goal": domain.goal
        })

    subdomains = subdomain_generator(domain_info)
    for domain_subdomains in subdomains:
        matching_domain = next((d for d in domains if d.value == domain_subdomains['domain']), None)
        if matching_domain:
            topic_info.append({
                "domain": domain_subdomains['domain'],
                "subdomain": domain_subdomains['subdomain'],
                "description": domain_subdomains['description'],
                "goal": domain_subdomains['goal']
            })

    topics = topic_generator(topic_info)

    subdomains_path = os.path.join(os.path.dirname(__file__), "generation_datasets", "subdomains.jsonl")
    subdomains.to_json(subdomains_path)

    topics_path = os.path.join(os.path.dirname(__file__), "generation_datasets", "topics.jsonl")
    topics.to_json(topics_path)

if __name__ == "__main__":
    main()