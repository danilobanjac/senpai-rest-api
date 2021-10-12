import json
import string
import subprocess
from contextlib import contextmanager

import spacy


sample_rate = 16000

TRANSLATION_TABLE = str.maketrans("", "", string.whitespace)


@contextmanager
def add_pipe(nlp, pipe_name: str):
    """Add certain pipe upon enterting the context and remove it on the exit."""

    nlp.add_pipe(pipe_name)

    try:
        yield
    finally:
        nlp.remove_pipe(pipe_name)


def highlight_key_phrases(nlp, text: str) -> str:
    """Highlight the key phrases in the given text."""

    highlighted_document = text
    document = nlp(text)
    phrases = spacy.util.filter_spans(
        [phrase.chunks[0] for phrase in document._.phrases if phrase.rank >= 0.05]
    )
    document.ents = []
    document.ents = phrases
    document_settings = document.to_json()
    entities = document_settings["ents"]

    for (index, ent) in enumerate(entities, start=0):
        skip = index * 4
        highlighted_document = (
            highlighted_document[0 : ent["start"] + skip]
            + f"**{highlighted_document[ent['start'] + skip: ent['end'] + skip]}**"
            + highlighted_document[ent["end"] + skip :]
        )

    return highlighted_document


def summarize_text(nlp, text: str) -> str:
    """Summarize the given text."""

    result = " ".join(
        token.text for token in nlp(text)._.textrank.summary(preserve_order=True)
    )

    if text.translate(TRANSLATION_TABLE) == result.translate(TRANSLATION_TABLE):
        return text

    return result


def transcribe_audio(file_name: str, recognizer) -> str:
    """Transcibe the given audio file using a proper recognizer."""

    results = []
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            file_name,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
    )

    while audio_data := process.stdout.read(4000):
        if recognizer.AcceptWaveform(audio_data):
            results.append(json.loads(recognizer.Result())["text"])

    results.append(json.loads(recognizer.FinalResult())["text"])

    result = " ".join(results)

    if result.isspace():
        return ""

    return result
