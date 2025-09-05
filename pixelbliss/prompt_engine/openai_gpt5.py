import os
import random
import asyncio
from typing import List, Optional, Dict
from openai import OpenAI, AsyncOpenAI
from .base import PromptProvider
from .knobs import KnobSelector

class OpenAIGPT5Provider(PromptProvider):
    """OpenAI GPT-5 implementation of the PromptProvider protocol."""
    
    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the OpenAI GPT-5 provider.
        
        Args:
            model: The OpenAI model to use. Defaults to "gpt-5".
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def make_base_with_knobs(self, base_knobs: Dict[str, str], avoid_list: List[str] = None) -> str:
        """
        Generate a base prompt using the new knobs system.
        
        Args:
            base_knobs: Dictionary containing selected values for each base knob category
            avoid_list: List of elements to avoid in the generated prompt
            
        Returns:
            str: Generated base prompt incorporating all knob values
        """
        # Format the knobs into a structured prompt
        knobs_description = []
        for knob_name, knob_value in base_knobs.items():
            knobs_description.append(f"{knob_name}: {knob_value}")
        
        avoid_text = ""
        if avoid_list:
            avoid_text = f" Avoid: {', '.join(avoid_list)}."
        
        system_prompt = (
            "You are a creative AI that generates aesthetic image prompts for wallpaper art using specific aesthetic control knobs. "
            "Create prompts that are highly visual, artistic, very detailed, and suitable for AI image generation. "
            "They should induce aesthetic and eudaimonic pleasure. "
            "Incorporate ALL the provided knob values seamlessly into a cohesive, beautiful prompt. "
            "Rules: No real people, no logos, no NSFW content. Include negative prompts for quality control."
        )
        
        user_prompt = (
            f"Generate a detailed, creative aesthetic wallpaper prompt incorporating these specific aesthetic elements:\n\n"
            f"{chr(10).join(knobs_description)}\n\n"
            f"Create a cohesive, beautiful image description that seamlessly blends all these elements.{avoid_text}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    def make_variants_with_knobs(self, base_prompt: str, k: int, variant_knobs_list: List[Dict[str, str]], avoid_list: List[str] = None) -> List[str]:
        """
        Generate k variations of a base prompt using the new knobs system.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            variant_knobs_list: List of k dictionaries, each containing variant knob values
            avoid_list: List of elements to avoid in the generated prompts
            
        Returns:
            List[str]: List of k prompt variations incorporating knob values
        """
        avoid_text = ""
        if avoid_list:
            avoid_text = f" Avoid: {', '.join(avoid_list)}."
        
        system_prompt = (
            "You are a creative AI that generates variations of image prompts using specific aesthetic control knobs. "
            "Given a base prompt and variant knobs (tone_curve, color_grade, surface_fx), create variations that "
            "maintain the core identity of the base prompt while applying the specified aesthetic modifications. "
            "The variant knobs should subtly modify the mood and visual treatment without changing the fundamental subject. "
            "Keep them aesthetic and wallpaper-friendly. "
            "Rules: No real people, no logos, no NSFW. Include negative prompts for quality control."
        )
        
        variants = []
        for i, variant_knobs in enumerate(variant_knobs_list):
            # Format the variant knobs
            knobs_description = []
            for knob_name, knob_value in variant_knobs.items():
                knobs_description.append(f"{knob_name}: {knob_value}")
            
            user_prompt = (
                f"Base prompt: {base_prompt}\n\n"
                f"Apply these aesthetic modifications:\n"
                f"{chr(10).join(knobs_description)}\n\n"
                f"Generate one variation that maintains the core identity while applying these aesthetic treatments.{avoid_text}"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            variant = response.choices[0].message.content.strip()
            variants.append(variant)
        
        return variants

    def make_alt_text(self, base_prompt: str, variant_prompt: str) -> str:
        """
        Generate alt text for an image based on prompts using OpenAI GPT-5.
        
        Args:
            base_prompt: The original base prompt used for image generation.
            variant_prompt: The specific variant prompt used for the final image.
            
        Returns:
            str: Concise, descriptive alt text for accessibility.
        """
        system_prompt = (
            "You are an AI that generates concise, descriptive alt text for images. "
            "Alt text should be 1-2 sentences, describing the visual elements without mentioning AI generation."
        )
        user_prompt = f"Generate alt text for an image based on this prompt: {variant_prompt}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=100  # Shorter for alt text
        )
        return response.choices[0].message.content.strip()

    async def make_variants_with_knobs_async(self, base_prompt: str, k: int, variant_knobs_list: List[Dict[str, str]], avoid_list: List[str] = None, max_concurrency: Optional[int] = None, progress_logger=None) -> List[str]:
        """
        Generate k variations of a base prompt using knobs system asynchronously.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            variant_knobs_list: List of k dictionaries, each containing variant knob values
            avoid_list: List of elements to avoid in the generated prompts
            max_concurrency: Maximum number of concurrent API calls. None means no limit.
            progress_logger: Optional progress logger for tracking generation progress.
            
        Returns:
            List[str]: List of k prompt variations with knobs applied.
        """
        # Set up concurrency control
        if max_concurrency is None:
            max_concurrency = k
        
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        
        # Start progress tracking
        if progress_logger:
            progress_logger.start_operation("prompt_generation", k, "parallel prompt generation")
        
        avoid_text = ""
        if avoid_list:
            avoid_text = f" Avoid: {', '.join(avoid_list)}."
        
        system_prompt = (
            "You are a creative AI that generates variations of image prompts using specific aesthetic control knobs. "
            "Given a base prompt and variant knobs (tone_curve, color_grade, surface_fx), create variations that "
            "maintain the core identity of the base prompt while applying the specified aesthetic modifications. "
            "The variant knobs should subtly modify the mood and visual treatment without changing the fundamental subject. "
            "Keep them aesthetic and wallpaper-friendly. "
            "Rules: No real people, no logos, no NSFW. Include negative prompts for quality control."
        )
        
        # Create wrapper function that updates progress
        async def generate_single_variant_with_progress(index: int, variant_knobs: Dict[str, str]):
            try:
                # Format the variant knobs
                knobs_description = []
                for knob_name, knob_value in variant_knobs.items():
                    knobs_description.append(f"{knob_name}: {knob_value}")
                
                user_prompt = (
                    f"Base prompt: {base_prompt}\n\n"
                    f"Apply these aesthetic modifications:\n"
                    f"{chr(10).join(knobs_description)}\n\n"
                    f"Generate one variation that maintains the core identity while applying these aesthetic treatments.{avoid_text}"
                )

                if semaphore:
                    async with semaphore:
                        response = await self.async_client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                        )
                else:
                    response = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                    )
                
                result = response.choices[0].message.content.strip()
                if progress_logger:
                    progress_logger.update_operation_progress("prompt_generation")
                return result
            except Exception as e:
                if progress_logger:
                    progress_logger.update_operation_progress("prompt_generation")
                raise e
        
        # Create tasks for all variants
        tasks = [
            asyncio.create_task(generate_single_variant_with_progress(i, variant_knobs))
            for i, variant_knobs in enumerate(variant_knobs_list)
        ]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        variants = []
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log error but continue with other variants
                # Use a fallback variant with knobs applied
                variant_knobs = variant_knobs_list[i]
                knobs_text = ", ".join([f"{knob}: {value}" for knob, value in variant_knobs.items()])
                fallback_variant = f"{base_prompt} with {knobs_text}"
                variants.append(fallback_variant)
                failed_count += 1
            else:
                variants.append(result)
        
        # Finish progress tracking
        if progress_logger:
            success = failed_count == 0
            progress_logger.finish_operation("prompt_generation", success)
            if failed_count > 0:
                progress_logger.warning(f"{failed_count} prompt variants used fallback generation")
        
        return variants
