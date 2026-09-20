"""
Perturbation Engine for PromptBench.
Handles multi-cultural demographic banks, scenario templates, and combinatorial slot-filling.
"""

from .name_banks import (
    CULTURAL_NAME_BANKS,
    PRONOUN_MAP,
    SES_INDICATORS,
    NATIONALITY_BANKS,
    generate_demographic_profiles,
)
from .templates import (
    SCENARIO_TEMPLATES,
    BasePromptTemplate,
    get_template_by_id,
    get_templates_by_category,
)
from .engine import PerturbationEngine

__all__ = [
    "CULTURAL_NAME_BANKS",
    "PRONOUN_MAP",
    "SES_INDICATORS",
    "NATIONALITY_BANKS",
    "generate_demographic_profiles",
    "SCENARIO_TEMPLATES",
    "BasePromptTemplate",
    "get_template_by_id",
    "get_templates_by_category",
    "PerturbationEngine",
]
