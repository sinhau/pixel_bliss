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

# PixelBliss — richly diverse, balanced controls
# Any random combo should feel distinct yet still spark joy, calm, and aesthetic wonder.

VIBE = [
    # light / airy / bright calm
    "serene", "gentle", "soothing", "dreamlike", "zen", "harmonious", "uplifting", "quiet awe",
    "warm nostalgia", "blue-hour hush", "golden-hour calm", "moonlit wonder", "meditative", "tranquil glow",
    "vibrant joy", "playful spark", "lush immersion", "mystical calm", "cosmic wonder", "radiant serenity",
    "crisp alpine clarity", "ocean-breath ease", "forest-bath tranquility", "garden bloom serenity",
    "rain-washed renewal", "misty dawn hush", "twilight reverie", "starlight drift", "lantern-lit coziness",
    "sun-dappled warmth", "prismatic calm", "crystalline awe", "ancient reverie", "subtle grandeur",
    "floating tranquility", "weightless lift", "breezy optimism", "whimsical calm", "soft jubilance",
    "sacred stillness", "tidal lull", "aurora reverie", "sakura hush", "fjord quiet", "tropical ease",
    "snow-day hush", "fireplace glow", "otherworldly gentleness",
    # darker / deeper calm & wonder
    "nocturne calm", "midnight sanctuary", "inkwell serenity", "cathedral dusk", "stormwatch stillness",
    "emberside reverie", "shadowed wonder", "deep-sea hush", "subterranean hush", "eclipse awe",
    "rain-on-asphalt calm", "forest-night hush", "moonshadow ease", "afterglow hush", "duskscape wonder",
    "midnight garden", "obsidian calm", "lantern-in-fog", "grotto quiet", "night-bloom jasmine calm",
    "nebula night drift", "noir serenity",
    # novel / evocative calm
    "lunar lullaby", "astral cradle", "harbor dusk ease", "tea-house hush", "temple bell hush",
    "monsoon veranda calm", "cavern emberglow", "subglacial stillness", "equinox balance",
    "zephyr ease", "river-stone composure", "stargazer’s rest", "hammock afternoon", "shadow garden",
    "gentle eclipse", "polar midnight glow", "copper-lantern warmth", "incense-and-ink quiet"
]

PALETTE = [
    # monochrome families (light to dark)
    "monochrome powder blue", "monochrome fog gray", "monochrome warm ivory", "monochrome cool slate",
    "monochrome jade", "monochrome indigo", "monochrome sand", "monochrome blush", "monochrome dusk plum",
    "monochrome lagoon teal", "monochrome charcoal", "monochrome obsidian",
    # analogous flows
    "analogous seafoam→mint", "analogous lilac→periwinkle", "analogous sand→terracotta",
    "analogous teal→indigo", "analogous coral→apricot", "analogous moss→sage",
    "analogous gold→umber", "analogous ice blue→mint", "analogous aubergine→plum",
    # complementary (soft + dark)
    "soft complementary blush↔sage", "soft complementary peach↔teal", "soft complementary lavender↔honey",
    "soft complementary apricot↔cobalt", "soft complementary coral↔emerald", "soft complementary plum↔olive",
    "deep complementary oxblood↔eucalyptus", "midnight teal↔amber",
    # split / triads (gentle)
    "split-comp sunflower⇢teal+plum", "split-comp seafoam⇢coral+lilac", "soft triad mint/peach/lavender",
    # duotone & accent
    "duotone blush/gold", "duotone jade/ivory", "duotone navy/pearl", "duotone indigo/sand",
    "duotone charcoal/rose-gold", "duotone obsidian/brass", "duotone plum/smoke",
    # gradients & spectral
    "sunrise gradient peach/lilac", "sunset sorbet coral/apricot/rose", "aurora gradient teal/violet",
    "glacier gradient aqua/steel", "tide gradient seafoam/blue", "ember gradient orange/crimson",
    "opal gradient white/green/pink", "prism sweep full-spectrum (muted)", "neon lullaby cyan/pink (soft)",
    "moonlit indigo→brass", "ink→gold fleck fade",
    # earth / mineral / botanical
    "earthy neutrals oat/stone/moss", "savanna straw/khaki/bronze", "canyon rust/clay/mesa",
    "fjord slate/fog/ice", "meadow clover/fern/dew", "orchard pear/apple/blossom",
    "lapis & sandstone", "malachite & mint", "rose-quartz & copper", "obsidian & ember",
    "terracotta & turquoise", "sandstone & azure", "eucalyptus silver/green", "tea-stain sepia",
    "charcoal & honey", "meteorite iron/rust/smoke", "copper patina verdigris/umber",
    # oceanic / sky
    "reef coral/turquoise/foam", "deep sea indigo/cyan", "stormcloud gray/indigo",
    "iceberg blue/gray", "midnight navy/silver", "moonstone gray/opal",
    # jewel / festival
    "jewel tones sapphire/emerald/amethyst", "peacock teal/emerald/plum", "citrine gold/amber",
    "garnet/wine/rosewood", "labradorite blue/gray/black", "black opal iridescence",
    # luminous / iridescent
    "butterfly wing iridescence", "mother-of-pearl shimmer", "firefly glow green/gold",
    "holographic pearl mist", "soap-film spectral",
    # seasonal
    "autumn maple/rust/gold", "spring lilac/mint/cream", "winter navy/silver/frost",
    "midsummer fern/moss/sky", "monsoon teal/ash",
    # flora & night variants
    "lotus pond teal/pink/gold", "orchid magenta/lavender", "desert bloom cactus green/rose",
    "sakura night magenta/navy", "lantern amber/indigo",
    # sun-kissed & dusk
    "sun-bleached coral/turquoise", "mango sunset orange/yellow", "honey amber/ivory",
    "dusky rose/blackcurrant", "smoky quartz/graphite",
    # deep / nocturne sets
    "ink/charcoal/graphite", "obsidian/onyx", "aubergine/plum/black", "midnight teal/black",
    "forest night green/black", "navy/black/smoke", "gunmetal/pewter/steel", "cinder/ember",
    "eclipse indigo/black", "stormy sea green/indigo/black"
]

LIGHT = [
    # bright / gentle
    "high-key airy diffused", "high-key luminous mist", "mid-key balanced bounce", "overcast softbox",
    "golden-hour backlight haze", "golden-hour rim glow", "blue-hour side light",
    "fog-diffused glow", "volumetric rays (gentle)", "subsurface scattering (soft)",
    "water caustic ripples", "iridescent highlights", "pearlescent sheen", "translucent bloom",
    "starlight twinkle highlights", "candlelit amber glow", "lantern warmth", "polar shimmer haze",
    "tropical midday clarity", "holographic scatter", "rain-washed reflections", "glacier sparkle",
    # nocturne / dramatic (calm-leaning)
    "moonlit shimmer low-key", "nocturne low-key", "chiaroscuro rim glow", "candlelit tenebrism",
    "eclipse backlight", "moonshadow penumbra", "lantern pools", "neon dusk radiance",
    "starlight speckle", "night rain reflections", "embers underglow", "gilded lowlight",
    # novel natural phenomena
    "alpenglow ridge", "moonbow arc", "sundog ice halo", "nacreous cloud shimmer",
    "bioluminescent shoreline", "glowworm ceiling", "cathedral light shafts",
    "stained-glass caustics", "lighthouse sweep haze", "snowlight bounce",
    "mirage shimmer", "heat-haze wavering", "cavern oculus beam", "milky way haze"
]

COMPOSITION = [
    "negative space emphasis", "rule of thirds", "central symmetry", "radial symmetry", "golden spiral",
    "balanced asymmetry", "asymmetric anchor & drift", "S-curve meander", "diagonal sweep",
    "zigzag rhythm", "ring/halo focus", "nested portals", "frames within frames",
    "triptych panels", "stacked terraces", "floating cluster", "constellation grid",
    "isometric grid balance", "lattice tessellation", "wave band layers", "spiral cascade",
    "mandala bloom", "horizon split symmetry", "low horizon minimal", "high horizon breadth",
    "leading lines converge", "off-center focal bloom", "orbiting satellites",
    "corner-to-center pathway", "woven interlace",
    # novel but readable
    "archipelago scatter", "petal rosette radial", "portal diptych", "nested crescents",
    "braid helix path", "pendulum arc balance", "tilted horizon drift", "veil-layer depth",
    "labyrinth meander", "tide-line stratification"
]

TEXTURE = [
    # soft & paper
    "matte paper", "hot-press watercolor", "cold-press watercolor", "soft grain film",
    "linen weave", "cotton canvas", "washi fibers", "vellum smooth", "handmade paper deckle",
    # glass & ceramic
    "smooth glass", "frosted glass", "opaline milk glass", "porcelain glaze", "crackle glaze",
    "ceramic glaze pooling", "enamel shine", "resin gloss", "sea-glass frost",
    # fabric & tactile
    "silky", "satin", "velvet", "suede", "felted wool", "bouclé loop", "knit rib",
    "rattan weave", "bamboo slats", "cork fine grain", "tatami reed mat",
    # stone & organic
    "weathered stone", "polished marble", "basalt micro-pits", "terrazzo fleck",
    "sand dune ripples", "pebble smooth", "driftwood grain", "moss cushion",
    "lichen fleck", "mica micro-shimmer", "mother-of-pearl", "salt-crystal frost",
    "birch bark", "tufa porous", "beeswax polish",
    # light & particulate
    "bokeh texture", "pearlescent sheen", "iridescent highlights", "pollen dust", "rain enamel",
    "smoke wash"
]

STYLE = [
    # painterly & drawing
    "watercolor wash", "gouache matte", "oil glazing (soft)", "oil impasto", "pastel chalk",
    "graphite sketch", "colored pencil bloom", "charcoal sketch", "ink line & wash", "sumi-e ink wash",
    # print & graphic
    "vector minimal", "flat poster print", "risograph grain", "screenprint layers",
    "linocut relief", "woodblock print", "hand-etched engraving", "stipple illustration",
    "pointillism dots", "halftone microdots", "isoline topography",
    # craft & textile
    "paper cutout", "papercraft quilling", "kirigami folds", "collage layers",
    "embroidery stitch", "loom tapestry", "ikat weave motif", "batik resist", "shibori indigo dye",
    # glass, tile, inlay
    "stained glass", "leadlight mosaic", "ceramic tile mosaic", "terrazzo mosaic",
    "pietra dura inlay", "kintsugi-inspired cracks", "marquetry wood inlay",
    # contemporary digital & generative
    "digital painting (soft)", "stylized 3D", "low-poly isometric", "voxel miniature",
    "wireframe holography", "flow-field generative", "voronoi mosaic", "delaunay facets",
    "fractal geometry art", "soft neon gradient glow", "holographic render",
    "ray-marched fog", "metabloom organic shapes", "moiré interference patterns",
    # photographic & light
    "long-exposure light trails", "bokeh abstraction", "tilt-shift miniature", "cyanotype botanical",
    "silver gelatin print", "wet plate collodion", "lumen print photogram",
    # historical & decorative
    "fresco mural texture", "art nouveau flow", "retro poster print", "illuminated manuscript",
    # nature-material aesthetics
    "geode crystal slice", "sand art pattern", "marbled ink swirl", "clay stop-motion look",
    "oxidized bronze patina", "burnished metal relief", "cathedral stained glass (night)",
    # ethereal & ambient (light and dark)
    "bioluminescent glow art", "dreamlike pastel surrealism", "soft vaporwave haze", "gentle cyber-aesthetic",
    "chiaroscuro painting", "tenebrism still life", "ink nocturne", "shadow puppet silhouette",
    "neon noir glow", "smoke calligraphy", "star map linework", "nightscape long exposure",
    # novel, whimsical-yet-calm
    "petal tessellation", "glacier lace filigree", "silk-cloud drapery", "aurora thread embroidery",
    "paper-lantern lattice", "tea-stain botanical print", "stone-garden raking lines"
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

# PixelBliss — Final treatment controls (augmented for light + dark balance, novelty, and calm wonder)

TONE_CURVE = [
    # high-key / airy
    "high-key airy matte", "high-key luminous gloss", "high-key pastel wash",
    "high-key porcelain flat", "high-key soft-shoulder rolloff",
    # mid-key / balanced
    "mid-key balanced soft S-curve", "mid-key film-matte (lifted blacks)",
    "mid-key pearly radiance", "mid-key crisp neutral S-curve",
    "mid-key gentle log (filmic)", "mid-key soft bleach-bypass",
    "mid-key low-contrast pastel matte",
    # low-key / deep (calm-leaning)
    "low-key velvet deep", "low-key misty hush", "low-key glow (soft highlights)",
    "low-key noir matte (lifted toe)", "low-key obsidian rich blacks",
    "low-key cathedral contrast (soft shoulder)",
    # dusk / night tonality
    "dusk soft gamma", "moonlit gentle contrast", "eclipse curve (compressed highlights)",
    "aurora soft shoulder", "inkwell deep matte (soft toe)",
    # cinematic & photographic
    "balanced S-curve clean", "cinematic fade (lifted blacks, soft knee)",
    "contrast-lite micro-S", "silverprint curve (cool matte)",
    "tea-stain matte (warm lift)", "selenium curve (cool lift)",
    "platinum print curve (neutral soft)", "luma-glow mids (expanded midtones)",
    "subtle HDR serenity (protected highlights)", "soft roll-off highlights + cushioned shadows",
    "tonemapped hush (low contrast, high detail)", "velvet S-curve (low micro-contrast)"
]

COLOR_GRADE = [
    # neutral & classic warms/cools
    "neutral balanced", "warm gold + muted", "warm peach + gently saturated",
    "honey warm + pastels", "rosy warm + balanced", "sepia-cream nostalgic",
    "cool silver + desaturated", "cool platinum + muted", "cool blue + gently saturated",
    # pastels & soft jewels
    "mint–silver cool pastel", "lavender–peach pastel wash", "teal–mint cool + soft jewels",
    "jade–ivory clean neutral", "blush–gold radiant pastel", "pistachio–peach fresh",
    "fog gray–moonlight neutral cool",
    # gentle teal–orange family (muted)
    "teal–apricot dusk (muted)", "teal–amber twilight (soft)",
    # deep / nocturne harmonies
    "indigo–brass nocturne", "ink–gold dusk", "garnet–smoke wine dark",
    "midnight teal–black", "forest night green–ink", "navy–smoke steel",
    # minerals & patinas
    "copper–teal patina", "verdigris–umber aged copper", "charcoal + honey warm neutral",
    "meteorite iron + smoke", "labradorite blue–gray–black",
    # botanical & earth
    "sage–cream natural", "emerald–moss woodland", "moss–fog earthy",
    "sandstone–azure coastal", "desert clay + turquoise",
    # ocean / sky / glacier
    "ocean–foam clean cool", "glacier aqua + steel", "storm gray + indigo",
    # luminous & iridescent
    "holographic pearl mist", "soap-film spectral pastel", "black opal iridescent",
    "bioluminescent cyan–green on navy", "aurora magenta–teal",
    # moonlight & dawn/dusk moods
    "moonlight platinum + sage", "dawn peach + lilac haze", "dusk blue + rose",
    # floral & stone
    "orchid–moonstone cool-warm", "rose-quartz + copper", "eucalyptus silver + jade",
    # gentle split-tones
    "split-tone honey highlights / indigo shadows",
    "split-tone rose highlights / eucalyptus shadows",
    "split-tone peach highlights / cobalt shadows",
    "split-tone amber highlights / midnight-teal shadows",
    "split-tone copper highlights / obsidian shadows",
    "split-tone lavender highlights / charcoal shadows",
    # historical processes (softened)
    "cyanotype bluewash (soft)", "selenium cool-brown (gentle)",
    "platinum neutral-cool", "tea-sepia classic",
    # bright-yet-calm accents
    "pearl–blush luminous", "mint–smoke watercolor", "lantern amber + navy"
]

SURFACE_FX = [
    # clean & filmic base
    "crystal clean (no grain, crisp edges)", "clarity polish (soft sharpen)",
    "soft matte (light grain, soft edges, subtle vignette)",
    "film-lite (very light grain, faint vignette)", "film classic (fine grain, soft vignette)",
    "film grainy (medium soft grain)",
    # bloom / glow / diffusion
    "pearl glow (subtle bloom)", "dreamy bloom (medium bloom)",
    "halo shimmer (haloed highlights)", "haze bloom (soft haze + bloom)",
    "soft pro-mist diffusion", "film halation (subtle red halo)",
    # bokeh & focus play
    "bokeh field (background bokeh)", "creamy bokeh (circular)", "hex bokeh (gentle)",
    "star bokeh (soft 4-point)", "depth-of-field ramp (gradual falloff)",
    "tilt-shift miniature (subtle)",
    # glass / refraction / interference
    "frosted glass diffusion", "rain-on-glass refraction", "crystal refraction (soft shards)",
    "water ripple caustics (gentle)", "soap-film interference shimmer",
    "prismatic micro-flares", "dappled canopy light",
    # light atmosphere
    "god-rays beam overlay (subtle)", "mist gradient (top-down)", "depth fog (z-fade)",
    "snowlight sparkle (very subtle)", "embers underglow motes",
    # vignette & edges
    "feathered oval vignette", "iris vignette (soft center glow)", "edge burn (delicate)",
    "vintage frame fade",
    # tactile / timeworn (calm)
    "paper fiber speckle (subtle)", "pulp texture overlay (light)", "ink-bleed edges (soft)",
    "patina speckle (aged film)", "dust & scratches (fine, sparse)",
    # lens character
    "anamorphic streak (gentle)", "soft star filter (subtle)",
    "micro chromatic fringing", "radial blur bloom (subtle)",
    # seasonal / weather micro-FX
    "snow drift sparkle (subtle)", "dew sparkle micro-bokeh", "soft rainfall streaks",
    # artistic overlays
    "stained-glass caustics", "paper-lantern glow wash", "lumen wash (photo-print look)"
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
