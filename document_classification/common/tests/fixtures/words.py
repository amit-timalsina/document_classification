import pytest

from document_classification.common.schemas.word import Word


@pytest.fixture
def words():
    """Fixture of words for Line schema tests."""
    return [
        Word(text="Hello", x0=0, y0=0, x2=5, y2=10),
        Word(text="World", x0=6, y0=0, x2=10, y2=10),
        Word(text="Foo", x0=0, y0=10, x2=5, y2=20),
        Word(text="Bar", x0=6, y0=10, x2=10, y2=20),
    ]
