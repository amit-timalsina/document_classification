import pytest

from document_classification.common.exceptions.bounding_box_error import BoundingBoxError
from document_classification.common.schemas.bounding_box import BoundingBox


@pytest.fixture
def bounding_box():
    """Test fixture for BoundingBox."""
    return BoundingBox(x_min=0, y_min=0, x_max=5, y_max=5)


class TestBoundingBox:
    """Test suite for the BoundingBox class."""

    def test_center(self, bounding_box: BoundingBox):
        """Test that the center point is calculated correctly."""
        assert bounding_box.center == (2.5, 2.5)

    def test_width(self, bounding_box: BoundingBox):
        """Test that the width is calculated correctly."""
        expected_width = bounding_box.x_max - bounding_box.x_min
        assert bounding_box.width == expected_width

    def test_height(self, bounding_box: BoundingBox):
        """Test that the height is calculated correctly."""
        assert bounding_box.height == bounding_box.y_max - bounding_box.y_min

    def test_x_max_must_be_greater_than_or_equal_to_x_min(self):
        """Test that x_max is greater than x_min."""
        with pytest.raises(BoundingBoxError):
            BoundingBox(x_min=5, y_min=0, x_max=4, y_max=5)

    def test_y_max_must_be_greater_than_or_equal_to_y_min(self):
        """Test that y_max is greater than y_min."""
        with pytest.raises(BoundingBoxError):
            BoundingBox(x_min=0, y_min=5, x_max=5, y_max=4)
