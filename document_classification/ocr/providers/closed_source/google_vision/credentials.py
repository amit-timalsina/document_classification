import os
from base64 import b64decode
from pathlib import Path


class GoogleCredentialsManager:
    """Manages the Google Vision SDK credentials."""

    @staticmethod
    def setup_credentials() -> None:
        """
        Set up the Google Vision API credentials.

        If the env GOOGLE_APPLICATION_CREDENTIALS is not set, a RuntimeError is raised.
        If the value is a path to a JSON file, it is set as the value of the environment variable.
        If the value is a base64-encoded JSON string, it is decoded and written to a file named
            "key.json" in the same directory as this file, and the path to that file is set as the
            value of the env variable.
        """
        service_key_encoded = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not service_key_encoded:
            msg = "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
            raise RuntimeError(msg)

        if service_key_encoded.lower().endswith(".json"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_encoded
        else:
            service_key_decoded = b64decode(service_key_encoded).decode()
            key_file = Path(__file__).parent / "key.json"
            key_file.write_text(service_key_decoded)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file)
