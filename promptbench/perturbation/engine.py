"""
Combinatorial Perturbation Engine for PromptBench.
Performs systematic slot substitution while preserving 100% invariant substantive content.
"""

from typing import List, Dict, Tuple, Optional, Any
from ..core.types import DemographicProfile, PerturbedPrompt, DemographicAxis
from .templates import BasePromptTemplate, SCENARIO_TEMPLATES
from .name_banks import generate_demographic_profiles, CULTURAL_NAME_BANKS, PRONOUN_MAP, SES_INDICATORS


class PerturbationEngine:
    """
    Engine responsible for taking base prompt templates and synthesizing
    counterfactual, controlled prompt variations across demographic axes.
    """

    def __init__(self, templates: Optional[List[BasePromptTemplate]] = None):
        self.templates = templates or SCENARIO_TEMPLATES

    @staticmethod
    def render_prompt(template: BasePromptTemplate, profile: DemographicProfile) -> str:
        """
        Substitutes demographic slots into the template text.
        Algorithm:
            Replaces '{name}', '{nationality}', '{education}', '{neighborhood}',
            '{extracurricular}', '{pronoun_subject}', '{pronoun_object}', '{pronoun_possessive}'.
        """
        replacements = {
            "name": profile.name,
            "nationality": profile.nationality,
            "education": profile.education_institution or "State University (B.S. in Computer Science)",
            "neighborhood": profile.residence_neighborhood or "Metro Area",
            "extracurricular": profile.extracurricular_cue or "Volunteer and community member",
            "pronoun_subject": profile.pronoun_subject,
            "pronoun_object": profile.pronoun_object,
            "pronoun_possessive": profile.pronoun_possessive,
            "title": profile.attributes.get("title", ""),
        }
        
        rendered = template.template_text
        for slot, value in replacements.items():
            rendered = rendered.replace(f"{{{slot}}}", value)
        return rendered

    def generate_counterfactual_dataset(
        self,
        template_ids: Optional[List[str]] = None,
        names_per_group: int = 2,
    ) -> Tuple[List[PerturbedPrompt], List[Tuple[PerturbedPrompt, PerturbedPrompt]]]:
        """
        Generates a comprehensive dataset of perturbed prompts along with matched baseline pairs.
        
        Returns:
            - all_prompts: Complete list of generated PerturbedPrompt instances.
            - paired_prompts: List of (baseline_prompt, perturbed_prompt) tuples for paired statistical analysis.
        """
        selected_templates = (
            [t for t in self.templates if t.template_id in template_ids]
            if template_ids
            else self.templates
        )

        all_prompts: List[PerturbedPrompt] = []
        paired_prompts: List[Tuple[PerturbedPrompt, PerturbedPrompt]] = []

        profiles = generate_demographic_profiles(names_per_group=names_per_group)

        # Baseline reference profile (Canonical Anglo Male Middle-SES reference)
        baseline_profile = DemographicProfile(
            profile_id="dp_baseline_ref",
            name="Alex Morgan",
            gender="neutral",
            ethnicity="anglo_western",
            nationality="United States",
            ses_level="middle",
            pronoun_subject="they",
            pronoun_object="them",
            pronoun_possessive="their",
            education_institution="State University (B.S. in Computer Science)",
            residence_neighborhood="Suburban Metro Area",
            extracurricular_cue="University peer tutor and community coding participant",
            attributes={"culture_label": "Baseline Neutral", "ses_label": "Middle SES"},
        )

        for tmpl in selected_templates:
            # 1. Create canonical baseline prompt for this template
            base_rendered = self.render_prompt(tmpl, baseline_profile)
            base_prompt = PerturbedPrompt(
                prompt_id=f"{tmpl.template_id}_base",
                template_id=tmpl.template_id,
                scenario_category=tmpl.scenario_category,
                demographic_profile=baseline_profile,
                rendered_prompt=base_rendered,
                is_baseline=True,
                metadata={"axis": "baseline", "group": "reference_neutral"},
            )
            all_prompts.append(base_prompt)

            # 2. Generate controlled perturbations for each profile
            for prof in profiles:
                rendered = self.render_prompt(tmpl, prof)
                variant_prompt = PerturbedPrompt(
                    prompt_id=f"{tmpl.template_id}_{prof.profile_id}",
                    template_id=tmpl.template_id,
                    scenario_category=tmpl.scenario_category,
                    demographic_profile=prof,
                    rendered_prompt=rendered,
                    is_baseline=False,
                    metadata={
                        "ethnicity": prof.ethnicity,
                        "gender": prof.gender,
                        "nationality": prof.nationality,
                        "ses_level": prof.ses_level,
                    },
                )
                all_prompts.append(variant_prompt)
                paired_prompts.append((base_prompt, variant_prompt))

        return all_prompts, paired_prompts
