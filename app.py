import os
import sys
import json
import tempfile
import threading
from enum import Enum
from typing import List, Optional, Union
from shutil import copyfileobj
from uuid import uuid4

from utils import add_pipe, highlight_key_phrases, summarize_text, transcribe_audio

import spacy
import pytextrank
from vosk import Model, KaldiRecognizer, SetLogLevel
import sentry_sdk
from bottle import Bottle, LocalResponse, run, request, response
from sentry_sdk.integrations.bottle import BottleIntegration
from tinydb import TinyDB, Query
from marshmallow import ValidationError
from marshmallow_dataclass import dataclass

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SetLogLevel(-1)  # No logs

SUFFIX = -1

# Initialize Sentry
sentry_sdk.init(
    dsn="https://948529131f3249aa82e22be242a3a3bb@o1025871.ingest.sentry.io/5992306",
    integrations=[BottleIntegration()],
)

# Initialize Bottle app
app = Bottle()


class SupportedLanguages(Enum):
    """Supported languages of AI models."""

    ENGLISH = "en"
    GERMAN = "de"


# Lock
lock = threading.Lock()

# Initialize spacy for different languages
nlp_en = spacy.load("en_core_web_trf")
nlp_de = spacy.load("de_dep_news_trf")

# Initialize vosk for different languages
sample_rate = 16000
model_en = Model("model_en")
model_de = Model("model_de")

recognizer_en = KaldiRecognizer(model_en, sample_rate)
recognizer_de = KaldiRecognizer(model_de, sample_rate)

# Database
db = TinyDB("pull_notes/pull_notes.json")
User = Query()

# Database structure
@dataclass
class EventMetadata:
    """Holds the metadata of the occuring event."""

    url: Optional[str]
    favIconUrl: Optional[str]
    tabID: Optional[int]
    sessionID: Optional[str]
    tabTitle: Optional[str]


@dataclass
class Settings:
    """Holds the settings of the pull note."""

    language: SupportedLanguages


@dataclass
class PullNote:
    """Structure of the pull note."""

    uuid: str
    content: str
    tags: List[str]
    created: int
    updated: Optional[int]
    eventMetadata: Optional[EventMetadata]
    settings: Settings


@dataclass
class SenpAIUser:
    """Structure of the main object stored in NoSQL."""

    uuid: str
    pullNotes: List[PullNote]


@dataclass
class HighlightText:
    """Represents the request stucture of the 'highlight' endpoint."""

    text: str
    language: SupportedLanguages


class SummarizeText(HighlightText):
    """Represents the request stucture of the 'summarize' endpoint."""


@app.post("/api/v1/users/")
def users_create() -> SenpAIUser:
    """Create a new user and return it."""

    document_id = db.insert({"uuid": str(uuid4()), "pullNotes": []})

    return db.get(doc_id=document_id)


@app.get("/api/v1/users/<user_uuid>")
def users_get(user_uuid: str) -> Union[SenpAIUser, LocalResponse]:
    """Retrieve an existing user."""

    try:
        (user,) = db.search(User.uuid == user_uuid)
    except ValueError:
        response.status = 404
        response.body = json.dumps({"error": "Requested user object does not exist."})

        return response

    return user


@app.post("/api/v1/users/<user_uuid>")
def users_update(user_uuid: str) -> Union[SenpAIUser, LocalResponse]:
    """Update an existing user and return the updated instance."""

    user_data = request.json

    try:
        SenpAIUser.Schema().load(user_data)
    except ValidationError as error:
        response.status = 400
        response.body = json.dumps({"error": error.normalized_messages()})

        return response

    (document_id,) = db.update(user_data, User.uuid == user_uuid)

    return db.get(doc_id=document_id)


@app.post("/api/v1/highlight")
def highlight():
    """Highlight the key phrases in the given text."""

    request_data = request.json

    try:
        request_data = HighlightText.Schema().load(request_data)
    except ValidationError as error:
        response.status = 400
        response.body = json.dumps({"error": error.normalized_messages()})

        return response

    nlp = nlp_en if request_data.language == SupportedLanguages.ENGLISH else nlp_de

    with lock:
        with add_pipe(nlp, "positionrank"):
            highlighted_text = highlight_key_phrases(nlp, request_data.text)

    return {
        "result": highlighted_text,
    }


@app.post("/api/v1/summarize")
def summarize():
    """Summarize the given text."""

    request_data = request.json

    try:
        request_data = SummarizeText.Schema().load(request_data)
    except ValidationError as error:
        response.status = 400
        response.body = json.dumps({"error": error.normalized_messages()})

        return response

    nlp = nlp_en if request_data.language == SupportedLanguages.ENGLISH else nlp_de

    with lock:
        with add_pipe(nlp, "biasedtextrank"):
            summarized_text = summarize_text(nlp, request_data.text)

    return {
        "result": summarized_text,
    }


@app.post("/api/v1/transcribe")
def transcribe():
    """Transcribe the given audio file."""

    audio_language = SupportedLanguages[request.forms.get("language")]
    audio_file = request.files.get("file")
    _, audio_file_suffix = audio_file.filename.split(".")
    recognizer = (
        recognizer_en if audio_language == SupportedLanguages.ENGLISH else recognizer_de
    )
    transcription = ""

    try:
        with tempfile.NamedTemporaryFile(
            dir="audio", suffix=f".{audio_file_suffix}"
        ) as temp_audio_file:
            copyfileobj(audio_file.file, temp_audio_file)
            transcription = transcribe_audio(temp_audio_file.name, recognizer)
    finally:
        recognizer.Reset()

    return {"result": transcription}


if __name__ == "__main__":
    run(app=app, **dict(zip(["host", "port"], sys.argv[1:])))
