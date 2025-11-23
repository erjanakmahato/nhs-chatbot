#!/usr/bin/env python3
"""
Utility script to ingest YouTube video transcripts into the Pinecone index
used by the NHS medical chatbot. Transcripts are converted into LangChain
Documents, chunked, embedded with the same HuggingFace model, and stored in
the existing `medical-chatbot` Pinecone index so Gemini can answer questions
grounded in the video content.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from src.helper import download_hugging_face_embeddings, text_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch transcripts for one or more YouTube videos and ingest them "
            "into the configured Pinecone index."
        )
    )
    parser.add_argument(
        "videos",
        metavar="VIDEO",
        nargs="+",
        help="YouTube video URLs or raw IDs (11 characters).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Preferred transcript language (default: en).",
    )
    parser.add_argument(
        "--index-name",
        default="medical-chatbot",
        help="Pinecone index to store the chunks (default: medical-chatbot).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/youtube_transcripts",
        help="Directory to cache raw transcript text (default: data/youtube_transcripts).",
    )
    return parser.parse_args()


def normalize_video_id(raw_value: str) -> str:
    """Extract the canonical 11-char video id from a URL or raw string."""
    candidate = raw_value.strip()
    if not candidate:
        raise ValueError("Empty video value provided.")

    parsed = urlparse(candidate)
    if parsed.scheme and parsed.netloc:
        hostname = parsed.netloc.lower()
        if "youtu.be" in hostname:
            video_id = parsed.path.lstrip("/")
        else:
            query_params = parse_qs(parsed.query)
            video_id = query_params.get("v", [None])[0]
            if not video_id and "/shorts/" in parsed.path:
                video_id = parsed.path.split("/shorts/")[-1].split("/")[0]
    else:
        video_id = candidate

    if not video_id or len(video_id) != 11:
        raise ValueError(f"Could not determine video id from input '{raw_value}'.")
    return video_id


def fetch_transcript_text(video_id: str, language: str) -> str:
    """Download the transcript text for a video."""
    preferred_languages = [language]
    if language != "en":
        preferred_languages.append("en")

    entries = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
    parts: List[str] = []
    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        parts.append(" ".join(text.split()))

    transcript_text = " ".join(parts)
    if not transcript_text:
        raise ValueError(f"Transcript for video {video_id} is empty.")
    return transcript_text


def fetch_video_title(video_id: str) -> str | None:
    """Fetch the video title via YouTube oEmbed (no extra dependency required)."""
    oembed_url = (
        "https://www.youtube.com/oembed?"
        f"url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    req = Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("title")
    except Exception:
        return None


def ensure_index(pc: Pinecone, index_name: str, dimension: int = 384) -> None:
    if pc.has_index(index_name):
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def build_documents(video_ids: Sequence[str], language: str, cache_dir: Path) -> List[Document]:
    documents: List[Document] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    for raw_value in video_ids:
        try:
            video_id = normalize_video_id(raw_value)
        except ValueError as exc:
            print(f"⚠️  Skipping '{raw_value}': {exc}")
            continue

        try:
            transcript_text = fetch_transcript_text(video_id, language)
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as exc:
            print(f"⚠️  Transcript unavailable for {video_id}: {exc}")
            continue
        except CouldNotRetrieveTranscript as exc:
            print(f"⚠️  Could not retrieve transcript for {video_id}: {exc}")
            continue
        except Exception as exc:
            print(f"⚠️  Unexpected error fetching transcript for {video_id}: {exc}")
            continue

        title = fetch_video_title(video_id)
        cache_path = cache_dir / f"{video_id}.txt"
        cache_path.write_text(transcript_text, encoding="utf-8")

        metadata = {
            "source": f"https://youtu.be/{video_id}",
            "video_id": video_id,
            "title": title or "YouTube Transcript",
            "type": "youtube_transcript",
        }
        documents.append(Document(page_content=transcript_text, metadata=metadata))
        print(f"✓ Collected transcript for {video_id} ({len(transcript_text)} chars)")

    return documents


def annotate_chunks(chunks: Iterable[Document]) -> List[Document]:
    annotated: List[Document] = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata or {})
        metadata.setdefault("chunk_origin", "youtube")
        metadata["chunk_index"] = idx
        annotated.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )
    return annotated


def main() -> None:
    args = parse_args()
    load_dotenv()

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY is not configured. Please update your .env file.")

    documents = build_documents(args.videos, args.language, Path(args.data_dir))
    if not documents:
        print("No transcripts were ingested. Nothing to upload.")
        return

    text_chunks = text_split(documents)
    if not text_chunks:
        print("Transcript chunking produced no data.")
        return

    annotated_chunks = annotate_chunks(text_chunks)
    embeddings = download_hugging_face_embeddings()

    pc = Pinecone(api_key=pinecone_key)
    ensure_index(pc, args.index_name, dimension=384)

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=args.index_name,
        embedding=embeddings,
    )
    vector_store.add_documents(annotated_chunks)
    print(f"✅ Added {len(annotated_chunks)} chunks to index '{args.index_name}'.")


if __name__ == "__main__":
    main()

