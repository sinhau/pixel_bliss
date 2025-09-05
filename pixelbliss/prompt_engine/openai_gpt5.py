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
        # Format the knobs into a structured prompt with clear categorization
        knobs_sections = {
            "Emotional Vibe": base_knobs.get("vibe", ""),
            "Color Palette": base_knobs.get("palette", ""),
            "Lighting Quality": base_knobs.get("light", ""),
            "Surface Texture": base_knobs.get("texture", ""),
            "Visual Composition": base_knobs.get("composition", ""),
            "Artistic Style": base_knobs.get("style", "")
        }
        
        knobs_description = []
        for section_name, knob_value in knobs_sections.items():
            if knob_value:
                knobs_description.append(f"• {section_name}: {knob_value}")
        
        avoid_text = ""
        if avoid_list:
            avoid_text = f"\n\nSTRICTLY AVOID: {', '.join(avoid_list)}"
        
        system_prompt = (
            "You are PixelBliss, an expert AI prompt architect specializing in creating wallpaper art that induces profound visual pleasure, "
            "sparks joy, evokes calm, and inspires aesthetic wonder. Your mission is to generate images that serve as visual sanctuaries—"
            "spaces of beauty that uplift the human spirit and create moments of transcendent aesthetic experience.\n\n"
            
            "CORE AESTHETIC PHILOSOPHY:\n"
            "• Generate images that induce eudaimonic pleasure (deep, meaningful joy)\n"
            "• Create visual experiences that spark wonder and contemplative calm\n"
            "• Design compositions that feel like visual poetry—harmonious, balanced, emotionally resonant\n"
            "• Craft prompts that result in images people want to live with daily as wallpapers\n\n"
            
            "KNOB INTEGRATION MASTERY:\n"
            "You will receive 6 aesthetic control knobs that must be seamlessly woven into a cohesive vision:\n"
            "1. Emotional Vibe - The feeling and mood the image should evoke\n"
            "2. Color Palette - The specific color harmony and relationships\n"
            "3. Lighting Quality - The character and behavior of light in the scene\n"
            "4. Surface Texture - The tactile and material qualities\n"
            "5. Visual Composition - The spatial arrangement and visual flow\n"
            "6. Artistic Style - The rendering technique and aesthetic approach\n\n"
            
            "PROMPT CRAFTING EXCELLENCE:\n"
            "• Synthesize all knob values into a unified, poetic description\n"
            "• Use rich, evocative language that guides AI image generation\n"
            "• Include specific technical details for optimal rendering\n"
            "• Balance artistic vision with technical precision\n"
            "• Always include negative prompts for quality assurance\n\n"
            
            "ABSOLUTE CONSTRAINTS:\n"
            "• NO real people, celebrities, or identifiable individuals\n"
            "• NO logos, brands, text, or commercial elements\n"
            "• NO NSFW, violent, or disturbing content\n"
            "• Focus on timeless, universally beautiful subjects\n"
            "• Prioritize visual harmony over complexity"
        )
        
        user_prompt = (
            f"Create a masterful wallpaper prompt that weaves these aesthetic elements into a cohesive vision of beauty:\n\n"
            f"{chr(10).join(knobs_description)}\n\n"
            f"Generate a detailed, poetic prompt that will result in an image that induces visual pleasure, sparks joy, "
            f"evokes calm, and creates a sense of aesthetic wonder. The image should feel like a visual sanctuary—"
            f"something someone would choose as their daily wallpaper because it brings them peace and inspiration.{avoid_text}"
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
            avoid_text = f"\n\nSTRICTLY AVOID: {', '.join(avoid_list)}"
        
        system_prompt = (
            "You are PixelBliss Variant Master, specializing in creating aesthetic variations that preserve the soul of an image "
            "while exploring different emotional and visual treatments. Your goal is to maintain the core beauty and wonder of the "
            "original vision while applying subtle aesthetic modifications that enhance visual pleasure and emotional resonance.\n\n"
            
            "VARIANT PHILOSOPHY:\n"
            "• Preserve the essential identity and emotional core of the base image\n"
            "• Apply aesthetic treatments that enhance rather than overwhelm\n"
            "• Create variations that feel like different moods of the same beautiful moment\n"
            "• Maintain the wallpaper-worthy quality and visual sanctuary feeling\n\n"
            
            "VARIANT KNOB MASTERY:\n"
            "You will receive 3 variant control knobs that modify the visual treatment:\n"
            "1. Tone Curve - Controls the overall brightness, contrast, and tonal character\n"
            "2. Color Grade - Adjusts color temperature, saturation, and color relationships\n"
            "3. Surface FX - Applies finishing effects like grain, bloom, bokeh, or clarity\n\n"
            
            "VARIATION CRAFTING PRINCIPLES:\n"
            "• Seamlessly integrate variant knobs into the existing prompt structure\n"
            "• Maintain the poetic and evocative language of the original\n"
            "• Enhance specific aspects while preserving overall harmony\n"
            "• Keep technical modifications subtle and aesthetically motivated\n"
            "• Ensure each variation could stand alone as a beautiful wallpaper\n\n"
            
            "CONSISTENCY REQUIREMENTS:\n"
            "• NO changes to the core subject matter or composition\n"
            "• NO alterations to the fundamental artistic style\n"
            "• Focus on mood, atmosphere, and visual treatment only\n"
            "• Maintain the same level of detail and descriptive richness\n"
            "• Preserve all quality and constraint guidelines from the original"
        )
        
        variants = []
        for i, variant_knobs in enumerate(variant_knobs_list):
            # Format the variant knobs with clear categorization
            knobs_sections = {
                "Tonal Treatment": variant_knobs.get("tone_curve", ""),
                "Color Grading": variant_knobs.get("color_grade", ""),
                "Surface Effects": variant_knobs.get("surface_fx", "")
            }
            
            knobs_description = []
            for section_name, knob_value in knobs_sections.items():
                if knob_value:
                    knobs_description.append(f"• {section_name}: {knob_value}")
            
            user_prompt = (
                f"BASE VISION:\n{base_prompt}\n\n"
                f"AESTHETIC MODIFICATIONS TO APPLY:\n"
                f"{chr(10).join(knobs_description)}\n\n"
                f"Create a variation that preserves the core beauty and emotional essence while applying these aesthetic treatments. "
                f"The result should feel like the same beautiful moment captured with different visual processing—maintaining all "
                f"the wonder, calm, and visual pleasure of the original while exploring a new aesthetic mood.{avoid_text}"
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
            "You are PixelBliss Alt Text Specialist, creating accessible descriptions for aesthetic wallpaper images that "
            "induce visual pleasure, joy, calm, and wonder. Your alt text should capture the essential visual beauty and "
            "emotional essence of the image in a way that conveys the aesthetic experience to users who cannot see it.\n\n"
            
            "ALT TEXT PHILOSOPHY:\n"
            "• Describe the visual elements that create beauty and emotional impact\n"
            "• Capture the mood, atmosphere, and aesthetic qualities\n"
            "• Focus on what makes the image visually pleasing and calming\n"
            "• Use evocative but accessible language\n\n"
            
            "DESCRIPTION GUIDELINES:\n"
            "• Keep to 1-2 concise sentences (under 125 characters when possible)\n"
            "• Lead with the most visually striking or beautiful elements\n"
            "• Include color, lighting, and compositional details that create the aesthetic impact\n"
            "• Mention the artistic style or technique if it contributes to the beauty\n"
            "• Avoid technical jargon or AI generation references\n"
            "• Focus on the sensory and emotional experience the image provides\n\n"
            
            "EXAMPLES OF GOOD ALT TEXT:\n"
            "• 'Soft watercolor mountains in pastel blues and pinks with golden morning light creating a serene, dreamlike landscape.'\n"
            "• 'Delicate cherry blossoms floating on still water with gentle bokeh and warm spring sunlight filtering through.'\n"
            "• 'Minimalist geometric patterns in sage green and cream with soft shadows creating a calm, balanced composition.'"
        )
        
        user_prompt = (
            f"Create accessible alt text for a beautiful wallpaper image based on this prompt:\n\n"
            f"{variant_prompt}\n\n"
            f"Focus on the visual elements that make this image aesthetically pleasing, calming, and wonder-inducing. "
            f"Describe what someone would see that creates the sense of beauty and emotional resonance."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=150  # Slightly longer for more descriptive alt text
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
            avoid_text = f"\n\nSTRICTLY AVOID: {', '.join(avoid_list)}"
        
        system_prompt = (
            "You are PixelBliss Variant Master, specializing in creating aesthetic variations that preserve the soul of an image "
            "while exploring different emotional and visual treatments. Your goal is to maintain the core beauty and wonder of the "
            "original vision while applying subtle aesthetic modifications that enhance visual pleasure and emotional resonance.\n\n"
            
            "VARIANT PHILOSOPHY:\n"
            "• Preserve the essential identity and emotional core of the base image\n"
            "• Apply aesthetic treatments that enhance rather than overwhelm\n"
            "• Create variations that feel like different moods of the same beautiful moment\n"
            "• Maintain the wallpaper-worthy quality and visual sanctuary feeling\n\n"
            
            "VARIANT KNOB MASTERY:\n"
            "You will receive 3 variant control knobs that modify the visual treatment:\n"
            "1. Tone Curve - Controls the overall brightness, contrast, and tonal character\n"
            "2. Color Grade - Adjusts color temperature, saturation, and color relationships\n"
            "3. Surface FX - Applies finishing effects like grain, bloom, bokeh, or clarity\n\n"
            
            "VARIATION CRAFTING PRINCIPLES:\n"
            "• Seamlessly integrate variant knobs into the existing prompt structure\n"
            "• Maintain the poetic and evocative language of the original\n"
            "• Enhance specific aspects while preserving overall harmony\n"
            "• Keep technical modifications subtle and aesthetically motivated\n"
            "• Ensure each variation could stand alone as a beautiful wallpaper\n\n"
            
            "CONSISTENCY REQUIREMENTS:\n"
            "• NO changes to the core subject matter or composition\n"
            "• NO alterations to the fundamental artistic style\n"
            "• Focus on mood, atmosphere, and visual treatment only\n"
            "• Maintain the same level of detail and descriptive richness\n"
            "• Preserve all quality and constraint guidelines from the original"
        )
        
        # Create wrapper function that updates progress
        async def generate_single_variant_with_progress(index: int, variant_knobs: Dict[str, str]):
            try:
                # Format the variant knobs with clear categorization
                knobs_sections = {
                    "Tonal Treatment": variant_knobs.get("tone_curve", ""),
                    "Color Grading": variant_knobs.get("color_grade", ""),
                    "Surface Effects": variant_knobs.get("surface_fx", "")
                }
                
                knobs_description = []
                for section_name, knob_value in knobs_sections.items():
                    if knob_value:
                        knobs_description.append(f"• {section_name}: {knob_value}")
                
                user_prompt = (
                    f"BASE VISION:\n{base_prompt}\n\n"
                    f"AESTHETIC MODIFICATIONS TO APPLY:\n"
                    f"{chr(10).join(knobs_description)}\n\n"
                    f"Create a variation that preserves the core beauty and emotional essence while applying these aesthetic treatments. "
                    f"The result should feel like the same beautiful moment captured with different visual processing—maintaining all "
                    f"the wonder, calm, and visual pleasure of the original while exploring a new aesthetic mood.{avoid_text}"
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
