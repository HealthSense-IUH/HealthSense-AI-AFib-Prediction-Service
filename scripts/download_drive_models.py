"""Download and verify the V6 model pair before building the Docker image."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MODEL_SHA256 = "e27370c51c8bbaf624abcbdabae40af5528d28bb2bd9e9249d1d5bf9cbb9b77c"
MODELS_DIR = Path(__file__).resolve().parents[1] / "app" / "models"


def drive_file_id(secret_name: str) -> str:
    link = os.getenv(secret_name, "").strip()
    if not link:
        raise ValueError(f"Missing {secret_name}; configure it as a repository Actions secret.")
    parsed = urlparse(link)
    if parsed.scheme != "https" or parsed.hostname != "drive.google.com":
        raise ValueError(f"{secret_name} must be an HTTPS Google Drive file link.")
    match = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)(?:/view)?/?", parsed.path)
    file_id = match.group(1) if match else parse_qs(parsed.query).get("id", [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError(f"{secret_name} does not contain a valid Drive file ID.")
    return file_id


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_pair(model: Path, card: Path) -> None:
    metadata = json.loads(card.read_text(encoding="utf-8"))
    if metadata.get("sha256") != MODEL_SHA256:
        raise ValueError("Model card SHA-256 differs from the pinned V6 artifact.")
    if sha256(model) != MODEL_SHA256:
        raise ValueError("Downloaded model SHA-256 mismatch; Docker build must not continue.")


def download(file_id: str, destination: Path, session=None) -> None:
    print(f"Downloading {destination.name} from Google Drive", flush=True)
    if session is None:
        import gdown

        result = gdown.download(id=file_id, output=str(destination), quiet=True, use_cookies=False)
        if result is None:
            raise RuntimeError(
                "Drive download failed. Share both files with Anyone with the link (Viewer), "
                "or configure GOOGLE_DRIVE_CREDENTIALS and grant its service account read access."
            )
    else:
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        with session.get(
            url, params={"alt": "media", "supportsAllDrives": "true"}, stream=True, timeout=120
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Drive returned an empty file for {destination.name}.")


def main() -> None:
    model_id = drive_file_id("GOOGLE_DRIVE_MODEL_URL")
    card_id = drive_file_id("GOOGLE_DRIVE_MODEL_CARD_URL")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = MODELS_DIR / "v6_stacking.pkl"
    card = MODELS_DIR / "v6_stacking.json"
    if model.is_file() and card.is_file():
        validate_pair(model, card)
        print("Cached V6 model pair passed SHA-256 verification.")
        return

    session = None
    credentials_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS", "").strip()
    if credentials_json:
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_info(
            json.loads(credentials_json), scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        session = AuthorizedSession(credentials)

    try:
        with tempfile.TemporaryDirectory(dir=MODELS_DIR) as temporary:
            staged_card = Path(temporary) / card.name
            staged_model = Path(temporary) / model.name
            download(card_id, staged_card, session)
            metadata = json.loads(staged_card.read_text(encoding="utf-8"))
            if metadata.get("sha256") != MODEL_SHA256:
                raise ValueError("Model card does not match the pinned V6 artifact.")
            download(model_id, staged_model, session)
            validate_pair(staged_model, staged_card)
            staged_model.replace(model)
            staged_card.replace(card)
    finally:
        if session is not None:
            session.close()
    print("V6 model pair downloaded and verified in app/models.")


if __name__ == "__main__":
    main()
