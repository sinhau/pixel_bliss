import pytest
from PIL import Image
from unittest.mock import Mock
from pixelbliss.imaging.variants import crop_pad, make_wallpaper_variants
from pixelbliss.config import WallpaperVariant


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return Image.new('RGB', (1000, 800), color='red')


@pytest.fixture
def square_image():
    """Create a square image for testing."""
    return Image.new('RGB', (500, 500), color='blue')


@pytest.fixture
def tall_image():
    """Create a tall image for testing."""
    return Image.new('RGB', (400, 800), color='green')


@pytest.fixture
def wide_image():
    """Create a wide image for testing."""
    return Image.new('RGB', (1200, 400), color='yellow')


class TestCropPad:
    """Test cases for crop_pad function."""

    def test_crop_pad_wider_image(self, sample_image):
        """Test cropping a wider image to fit target aspect ratio."""
        # Original: 1000x800 (ratio 1.25), Target: 400x400 (ratio 1.0)
        result = crop_pad(sample_image, 400, 400)
        
        assert result.size == (400, 400)
        assert isinstance(result, Image.Image)

    def test_crop_pad_taller_image(self, tall_image):
        """Test cropping a taller image to fit target aspect ratio."""
        # Original: 400x800 (ratio 0.5), Target: 600x400 (ratio 1.5)
        result = crop_pad(tall_image, 600, 400)
        
        assert result.size == (600, 400)
        assert isinstance(result, Image.Image)

    def test_crop_pad_exact_ratio(self, square_image):
        """Test resizing when image already has correct aspect ratio."""
        # Original: 500x500 (ratio 1.0), Target: 300x300 (ratio 1.0)
        result = crop_pad(square_image, 300, 300)
        
        assert result.size == (300, 300)
        assert isinstance(result, Image.Image)

    def test_crop_pad_upscale(self, sample_image):
        """Test upscaling an image to larger dimensions."""
        # Original: 1000x800, Target: 2000x1600 (same ratio)
        result = crop_pad(sample_image, 2000, 1600)
        
        assert result.size == (2000, 1600)
        assert isinstance(result, Image.Image)

    def test_crop_pad_downscale(self, sample_image):
        """Test downscaling an image to smaller dimensions."""
        # Original: 1000x800, Target: 500x400 (same ratio)
        result = crop_pad(sample_image, 500, 400)
        
        assert result.size == (500, 400)
        assert isinstance(result, Image.Image)

    def test_crop_pad_wide_to_tall(self, wide_image):
        """Test converting wide image to tall format."""
        # Original: 1200x400 (ratio 3.0), Target: 300x600 (ratio 0.5)
        result = crop_pad(wide_image, 300, 600)
        
        assert result.size == (300, 600)
        assert isinstance(result, Image.Image)

    def test_crop_pad_small_dimensions(self, sample_image):
        """Test cropping to very small dimensions."""
        result = crop_pad(sample_image, 50, 50)
        
        assert result.size == (50, 50)
        assert isinstance(result, Image.Image)

    def test_crop_pad_large_dimensions(self, sample_image):
        """Test cropping to very large dimensions."""
        result = crop_pad(sample_image, 4000, 3000)
        
        assert result.size == (4000, 3000)
        assert isinstance(result, Image.Image)


class TestMakeWallpaperVariants:
    """Test cases for make_wallpaper_variants function."""

    def test_make_wallpaper_variants_single(self, sample_image):
        """Test creating a single wallpaper variant."""
        variant = Mock(spec=WallpaperVariant)
        variant.name = "desktop"
        variant.w = 1920
        variant.h = 1080
        
        variants_cfg = [variant]
        result = make_wallpaper_variants(sample_image, variants_cfg)
        
        assert len(result) == 1
        assert "desktop" in result
        assert result["desktop"].size == (1920, 1080)
        assert isinstance(result["desktop"], Image.Image)

    def test_make_wallpaper_variants_multiple(self, sample_image):
        """Test creating multiple wallpaper variants."""
        desktop_variant = Mock(spec=WallpaperVariant)
        desktop_variant.name = "desktop"
        desktop_variant.w = 1920
        desktop_variant.h = 1080
        
        mobile_variant = Mock(spec=WallpaperVariant)
        mobile_variant.name = "mobile"
        mobile_variant.w = 1080
        mobile_variant.h = 1920
        
        tablet_variant = Mock(spec=WallpaperVariant)
        tablet_variant.name = "tablet"
        tablet_variant.w = 1024
        tablet_variant.h = 768
        
        variants_cfg = [desktop_variant, mobile_variant, tablet_variant]
        result = make_wallpaper_variants(sample_image, variants_cfg)
        
        assert len(result) == 3
        assert "desktop" in result
        assert "mobile" in result
        assert "tablet" in result
        
        assert result["desktop"].size == (1920, 1080)
        assert result["mobile"].size == (1080, 1920)
        assert result["tablet"].size == (1024, 768)
        
        for variant_image in result.values():
            assert isinstance(variant_image, Image.Image)

    def test_make_wallpaper_variants_empty_list(self, sample_image):
        """Test creating variants with empty configuration list."""
        variants_cfg = []
        result = make_wallpaper_variants(sample_image, variants_cfg)
        
        assert len(result) == 0
        assert isinstance(result, dict)

    def test_make_wallpaper_variants_square_formats(self, sample_image):
        """Test creating square format variants."""
        square_variant = Mock(spec=WallpaperVariant)
        square_variant.name = "square"
        square_variant.w = 1000
        square_variant.h = 1000
        
        variants_cfg = [square_variant]
        result = make_wallpaper_variants(sample_image, variants_cfg)
        
        assert len(result) == 1
        assert "square" in result
        assert result["square"].size == (1000, 1000)

    def test_make_wallpaper_variants_ultrawide(self, sample_image):
        """Test creating ultrawide format variants."""
        ultrawide_variant = Mock(spec=WallpaperVariant)
        ultrawide_variant.name = "ultrawide"
        ultrawide_variant.w = 3440
        ultrawide_variant.h = 1440
        
        variants_cfg = [ultrawide_variant]
        result = make_wallpaper_variants(sample_image, variants_cfg)
        
        assert len(result) == 1
        assert "ultrawide" in result
        assert result["ultrawide"].size == (3440, 1440)

    def test_make_wallpaper_variants_preserve_original(self, sample_image):
        """Test that original image is not modified."""
        original_size = sample_image.size
        
        variant = Mock(spec=WallpaperVariant)
        variant.name = "test"
        variant.w = 800
        variant.h = 600
        
        variants_cfg = [variant]
        make_wallpaper_variants(sample_image, variants_cfg)
        
        # Original image should remain unchanged
        assert sample_image.size == original_size

    def test_make_wallpaper_variants_different_aspect_ratios(self, square_image):
        """Test creating variants with very different aspect ratios from source."""
        # Source is square (1:1), create wide (16:9) and tall (9:16) variants
        wide_variant = Mock(spec=WallpaperVariant)
        wide_variant.name = "wide"
        wide_variant.w = 1600
        wide_variant.h = 900
        
        tall_variant = Mock(spec=WallpaperVariant)
        tall_variant.name = "tall"
        tall_variant.w = 900
        tall_variant.h = 1600
        
        variants_cfg = [wide_variant, tall_variant]
        result = make_wallpaper_variants(square_image, variants_cfg)
        
        assert len(result) == 2
        assert result["wide"].size == (1600, 900)
        assert result["tall"].size == (900, 1600)
