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

    def make_base(self, category: str) -> str:
        """
        Generate a base prompt for the given category using OpenAI GPT-5.
        
        Args:
            category: The category/theme for the prompt.
            
        Returns:
            str: Generated base prompt optimized for aesthetic wallpaper generation.
        """
        system_prompt = (
            "You are a creative AI that generates aesthetic image prompts for wallpaper art. "
            "Create prompts that are highly visual, artistic, very detailed, and suitable for AI image generation. "
            "They should induce aesthetic and eudaimonic pleasure, so the overall style of the image should induce this. "
            "Rules: No real people, no logos, no NSFW content. Include negative prompts like 'blurry, low quality, text'."
        )
        user_prompt = f"Generate a detailed, creative prompt for an aesthetic {category} themed image suitable for a wallpaper. Adhere strictly to the provided system prompts."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content.strip()

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

    async def _generate_single_variant_async(self, base_prompt: str, art_style: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        """
        Generate a single prompt variant asynchronously.
        
        Args:
            base_prompt: The original prompt to create variations from.
            art_style: The art style to emphasize in this variant.
            semaphore: Optional semaphore to limit concurrent API calls.
            
        Returns:
            str: Generated prompt variant.
        """
        system_prompt = (
            "You are a creative AI that generates variations of image prompts. "
            "Given a base prompt and an art style, create variations by incorporating the specified art style "
            "while changing colors, compositions, or adding artistic elements. "
            "EMPHASIZE the art style heavily in your prompt - it should be a dominant characteristic of the image. "
            "Try to add some random thematic element that induces aesthetic and eudaimonic pleasure. "
            "Keep them aesthetic and wallpaper-friendly. "
            "Rules: No real people, no logos, no NSFW. Include negative prompts."
        )
        
        user_prompt = (
            f"Base prompt: {base_prompt}\n\n"
            f"Art style to emphasize: {art_style}\n\n"
            f"Generate one creative variation of this prompt that heavily emphasizes the {art_style} art style. "
            f"Make sure the {art_style} style is a prominent and defining characteristic of the image."
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
        
        return response.choices[0].message.content.strip()

    async def make_variants_from_base_async(self, base_prompt: str, k: int, art_styles: List[str] = None, max_concurrency: Optional[int] = None, progress_logger=None) -> List[str]:
        """
        Generate k variations of a base prompt in parallel using OpenAI GPT-5.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            art_styles: List of art styles to randomly select from for each variant.
            max_concurrency: Maximum number of concurrent API calls. None means no limit.
            progress_logger: Optional progress logger for tracking generation progress.
            
        Returns:
            List[str]: List of k prompt variations with different styles and elements.
        """
        # Set up concurrency control
        if max_concurrency is None:
            max_concurrency = k
        
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        
        # Start progress tracking
        if progress_logger:
            progress_logger.start_operation("prompt_generation", k, "parallel prompt generation")
        
        # Pre-select art styles for each variant to ensure deterministic results
        selected_styles = []
        for _ in range(k):
            selected_style = random.choice(art_styles) if art_styles else "artistic style"
            selected_styles.append(selected_style)
        
        # Create wrapper function that updates progress
        async def generate_with_progress(index: int, base_prompt: str, style: str, semaphore: Optional[asyncio.Semaphore]):
            try:
                result = await self._generate_single_variant_async(base_prompt, style, semaphore)
                if progress_logger:
                    progress_logger.update_operation_progress("prompt_generation")
                return result
            except Exception as e:
                if progress_logger:
                    progress_logger.update_operation_progress("prompt_generation")
                raise e
        
        # Create tasks for all variants
        tasks = [
            asyncio.create_task(generate_with_progress(i, base_prompt, style, semaphore))
            for i, style in enumerate(selected_styles)
        ]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        variants = []
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log error but continue with other variants
                # For now, we'll use a fallback variant
                fallback_variant = f"{base_prompt} in {selected_styles[i]} style"
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

    def make_variants_from_base(self, base_prompt: str, k: int, art_styles: List[str] = None) -> List[str]:
        """
        Generate k variations of a base prompt using OpenAI GPT-5.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            art_styles: List of art styles to randomly select from for each variant.
            
        Returns:
            List[str]: List of k prompt variations with different styles and elements.
        """
        system_prompt = (
            "You are a creative AI that generates variations of image prompts. "
            "Given a base prompt and an art style, create variations by incorporating the specified art style "
            "while changing colors, compositions, or adding artistic elements. "
            "EMPHASIZE the art style heavily in your prompt - it should be a dominant characteristic of the image. "
            "Try to add some random thematic element that induces aesthetic and eudaimonic pleasure. "
            "Keep them aesthetic and wallpaper-friendly. "
            "Rules: No real people, no logos, no NSFW. Include negative prompts."
        )
        variants = []
        for _ in range(k):
            # Select a random art style for this variant
            selected_style = random.choice(art_styles) if art_styles else "artistic style"
            
            user_prompt = (
                f"Base prompt: {base_prompt}\n\n"
                f"Art style to emphasize: {selected_style}\n\n"
                f"Generate one creative variation of this prompt that heavily emphasizes the {selected_style} art style. "
                f"Make sure the {selected_style} style is a prominent and defining characteristic of the image."
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
