"""
Diverse control knobs for image generation prompt creation.

This module contains the comprehensive set of aesthetic control knobs for generating
base prompts and prompt variants, providing much more diversity than the previous
simple category/art_style system.
"""

from typing import Dict, List
import random

# ==========================
# BASE PROMPT KNOBS (6)
# ==========================

VIBE = [
    "serene", "gentle", "airy", "soothing", "glowing", "dreamlike", "zen",
    "harmonious", "weightless", "uplifting", "softly whimsical", "quiet awe",
    "warm nostalgia", "blue-hour hush", "golden-hour calm", "moonlit wonder",
    "meditative", "tranquil glow"
]

PALETTE = [
    "monochrome powder blue", "monochrome warm ivory", "monochrome cool slate",
    "analogous seafoam→mint", "analogous lilac→periwinkle", "analogous sand→terracotta",
    "soft complementary blush↔sage", "soft complementary peach↔teal",
    "duotone blush/gold", "duotone jade/ivory",
    "pastel sunrise peach/lavender/mint", "sunset sorbet coral/apricot/rose",
    "aurora gradient green/violet", "cotton-candy gradient pink/blue",
    "earthy neutrals oat/stone/moss", "jade & linen"
]

LIGHT = [
    "high-key airy diffused", "high-key luminous mist", "mid-key balanced bounce",
    "low-key velvet hush", "golden-hour backlight haze", "golden-hour rim glow",
    "blue-hour side light", "moonlit shimmer low-key", "overcast softbox",
    "fog-diffused glow", "volumetric rays (gentle)", "subsurface scattering (soft)",
    "water caustic ripples", "iridescent highlights", "pearlescent sheen",
    "translucent bloom"
]

TEXTURE = [
    "silky", "satin", "velvet", "matte paper", "hot-press watercolor",
    "cold-press watercolor", "oil impasto", "pastel chalk dust",
    "soft grain film", "bokeh texture", "smooth glass", "frosted glass",
    "porcelain glaze", "clay matte", "linen weave", "mica micro-shimmer"
]

COMPOSITION = [
    "negative space emphasis", "central symmetry", "radial symmetry",
    "balanced asymmetry", "golden spiral", "rule of thirds",
    "isometric grid balance", "diagonal flow", "ring composition",
    "clustered focal trio", "repetition & rhythm", "layered depth",
    "low horizon minimal", "top-down mandala", "floating scatter", "foreground frame"
]

STYLE = [
    "watercolor wash", "gouache matte", "oil impasto", "pastel chalk",
    "sumi-e ink wash", "airbrush smooth", "paper cutout", "origami-inspired",
    "stained glass", "ceramic glaze", "digital painting (soft)", "stylized 3D",
    "low-poly isometric", "vector minimal", "delicate line art", "bokeh abstraction"
]

# Base knobs dictionary for easy access
BASE_KNOBS = {
    "vibe": VIBE,
    "palette": PALETTE,
    "light": LIGHT,
    "texture": TEXTURE,
    "composition": COMPOSITION,
    "style": STYLE,
}

# ==========================
# VARIANT PROMPT KNOBS (3)
# ==========================

TONE_CURVE = [
    "high-key airy matte", "high-key luminous gloss", "high-key pastel wash",
    "mid-key balanced soft S-curve", "mid-key film-matte (lifted blacks)",
    "mid-key pearly radiance", "low-key velvet deep", "low-key misty hush",
    "low-key glow (soft highlights)", "pastel matte flat curve",
    "soft haze low-contrast", "balanced S-curve clean"
]

COLOR_GRADE = [
    "neutral balanced", "warm gold + muted", "warm peach + gently saturated",
    "honey warm + pastels", "rosy warm + balanced", "sepia-cream nostalgic",
    "cool silver + desaturated", "cool platinum + muted", "cool blue + gently saturated",
    "mint–silver cool pastel", "lavender–peach pastel wash", "teal–mint cool + soft jewels",
    "jade–ivory clean neutral", "blush–gold radiant pastel", "pistachio–peach fresh",
    "fog gray–moonlight neutral cool"
]

SURFACE_FX = [
    "crystal clean (no grain, crisp edges)", "soft matte (light grain, soft edges, subtle vignette)",
    "pearl glow (subtle bloom)", "dreamy bloom (medium bloom)", "halo shimmer (haloed highlights)",
    "film-lite (very light grain, faint vignette)", "film classic (fine grain, soft vignette)",
    "film grainy (medium soft grain)", "clarity polish (soft sharpen)", "haze bloom (soft haze + bloom)",
    "bokeh field (background bokeh)", "creamy bokeh (circular)", "hex bokeh (gentle)",
    "tilt-shift miniature (subtle)", "frosted glass diffusion", "pearlescent micro-shimmer"
]

# Variant knobs dictionary for easy access
VARIANT_KNOBS = {
    "tone_curve": TONE_CURVE,
    "color_grade": COLOR_GRADE,
    "surface_fx": SURFACE_FX,
}

# ==========================
# GUARDRAILS
# ==========================

AVOID = [
    "harsh clipping", "oversharpening", "excessive noise", "watermarks",
    "neon overload", "aggressive reds", "busy clutter", "posterization",
    "banding", "jarring high contrast", "text overlays"
]

# ==========================
# KNOB SELECTION FUNCTIONS
# ==========================

class KnobSelector:
    """Utility class for selecting knob values with various strategies."""
    
    @staticmethod
    def select_base_knobs(category: str = None) -> Dict[str, str]:
        """
        Select a complete set of base knobs for prompt generation.
        
        Args:
            category: Optional category hint (for future category-aware selection)
            
        Returns:
            Dict[str, str]: Dictionary with one value selected from each base knob category
        """
        return {
            knob_name: random.choice(knob_values)
            for knob_name, knob_values in BASE_KNOBS.items()
        }
    
    @staticmethod
    def select_variant_knobs() -> Dict[str, str]:
        """
        Select a complete set of variant knobs for prompt variation.
        
        Returns:
            Dict[str, str]: Dictionary with one value selected from each variant knob category
        """
        return {
            knob_name: random.choice(knob_values)
            for knob_name, knob_values in VARIANT_KNOBS.items()
        }
    
    
    @staticmethod
    def get_avoid_list() -> List[str]:
        """
        Get the list of elements to avoid in image generation.
        
        Returns:
            List[str]: List of elements to avoid
        """
        return AVOID.copy()
