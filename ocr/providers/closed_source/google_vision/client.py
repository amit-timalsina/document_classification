from google.cloud import vision


class GoogleVisionClient:
    """Client class for interacting with Google Vision API."""

    @staticmethod
    def create_client() -> vision.ImageAnnotatorClient:
        """Create a client for interacting with Google Vision API."""
        return vision.ImageAnnotatorClient()
