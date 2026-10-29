
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import shutil
import whisper
import uuid
import json

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build

from models import ReelDocument
from ocr import extract_text_from_video
from answer import answer_question
from vector_store import add_document


app = FastAPI(title="Personal Multimedia Knowledge Assistant")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_FILE = BASE_DIR / "playlists.json"
FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"

GOOGLE_CLIENT_FILE = BASE_DIR / "client_secret.json"
YOUTUBE_TOKEN_FILE = BASE_DIR / "youtube_token.json"
YOUTUBE_REDIRECT_URI = "http://127.0.0.1:8000/youtube/oauth/callback"

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly"
]

# Local, single-user prototype.
# Store each OAuth state's PKCE verifier until its callback arrives.
pending_oauth_states = {}


class YouTubeImportRequest(BaseModel):
    name: str
    playlist_url: str


class PlaylistCreate(BaseModel):
    name: str
    source: str = "instagram"


UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_playlists():
    if PLAYLISTS_FILE.exists():
        with open(PLAYLISTS_FILE, "r", encoding="utf-8") as file:
            playlists = json.load(file)
    else:
        playlists = []

    # Keep existing unassigned uploads accessible.
    if not any(p["id"] == "default" for p in playlists):
        playlists.insert(
            0,
            {
                "id": "default",
                "name": "My Videos",
                "created_at": datetime.now().isoformat(),
                "source": "local",
                "video_count": 0,
            },
        )
        save_playlists(playlists)

    return playlists


def save_playlists(playlists):
    with open(PLAYLISTS_FILE, "w", encoding="utf-8") as file:
        json.dump(playlists, file, ensure_ascii=False, indent=4)


def get_playlist(playlist_id):
    return next(
        (p for p in load_playlists() if p["id"] == playlist_id),
        None,
    )


def get_youtube_credentials():
    """Return valid saved credentials, or None if YouTube is not connected."""
    if not YOUTUBE_TOKEN_FILE.exists():
        return None

    try:
        credentials = Credentials.from_authorized_user_file(
            str(YOUTUBE_TOKEN_FILE),
            YOUTUBE_SCOPES,
        )

        if credentials.expired and credentials.refresh_token:
            credentials.refresh(GoogleRequest())
            YOUTUBE_TOKEN_FILE.write_text(
                credentials.to_json(),
                encoding="utf-8",
            )

        if not credentials.valid:
            return None

        return credentials

    except Exception as error:
        print(f"Could not load YouTube credentials: {error}")
        return None


def get_youtube_service():
    credentials = get_youtube_credentials()

    if not credentials:
        return None

    return build(
        "youtube",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def extract_playlist_id(value: str) -> str:
    """Accept a YouTube playlist URL or a raw playlist ID."""
    value = value.strip()

    if not value:
        raise HTTPException(
            status_code=400,
            detail="Enter a YouTube playlist URL.",
        )

    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()

        if hostname not in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Enter a YouTube playlist URL, "
                    "not an individual video URL."
                ),
            )

        playlist_id = parse_qs(parsed.query).get("list", [None])[0]

        if not playlist_id:
            raise HTTPException(
                status_code=400,
                detail="This URL does not contain a YouTube playlist ID.",
            )

        return playlist_id

    # Raw playlist IDs are accepted too.
    if "/" in value or "?" in value or "&" in value:
        raise HTTPException(
            status_code=400,
            detail="The playlist ID is not valid.",
        )

    return value


def fetch_youtube_transcript(video_id: str):
    """
    Try to retrieve an available transcript.

    Transcript retrieval is separate from YouTube OAuth. A transcript
    failure does not prevent importing a video's title and description.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:
            # Current youtube-transcript-api interface.
            result = YouTubeTranscriptApi().fetch(
                video_id,
                languages=["en", "hi"],
            )

            segments = [
                {
                    "text": snippet.text,
                    "start": float(snippet.start),
                    "end": float(snippet.start + snippet.duration),
                }
                for snippet in result.snippets
                if snippet.text.strip()
            ]

        except AttributeError:
            # Compatibility with older versions of the library.
            raw_segments = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=["en", "hi"],
            )

            segments = [
                {
                    "text": item["text"],
                    "start": float(item["start"]),
                    "end": float(item["start"] + item["duration"]),
                }
                for item in raw_segments
                if item.get("text", "").strip()
            ]

        transcript = " ".join(
            segment["text"] for segment in segments
        )

        return transcript, segments

    except Exception as error:
        print(
            f"Transcript unavailable for YouTube video {video_id}: {error}"
        )
        return "", []


print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded.")


@app.get("/")
def home():
    return {
        "status": "Personal Multimedia Knowledge Assistant is running."
    }


@app.get("/app")
def frontend():
    if not FRONTEND_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend file not found",
        )

    return FileResponse(FRONTEND_FILE)


@app.get("/playlists")
def list_playlists():
    return load_playlists()


@app.post("/playlists")
def create_playlist(playlist: PlaylistCreate):
    name = playlist.name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Playlist name cannot be empty",
        )

    playlists = load_playlists()

    if any(p["name"].casefold() == name.casefold() for p in playlists):
        raise HTTPException(
            status_code=409,
            detail="A playlist with that name already exists",
        )

    new_playlist = {
        "id": str(uuid.uuid4()),
        "name": name,
        "source": playlist.source,
        "created_at": datetime.now().isoformat(),
        "video_count": 0,
    }

    playlists.append(new_playlist)
    save_playlists(playlists)

    return new_playlist


@app.post("/playlists/{playlist_id}/upload")
async def upload_files(
    playlist_id: str,
    files: list[UploadFile] = File(...),
):
    if not get_playlist(playlist_id):
        raise HTTPException(
            status_code=404,
            detail="Playlist not found",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Select at least one video",
        )

    results = []

    for uploaded_file in files:
        original_name = Path(
            uploaded_file.filename or "video.mp4"
        ).name
        stored_name = f"{uuid.uuid4().hex}_{original_name}"
        file_path = UPLOAD_DIR / stored_name

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

            print(f"Processing: {original_name}")

            transcript_result = model.transcribe(str(file_path))
            ocr_text = extract_text_from_video(str(file_path))

            reel = ReelDocument(
                id=str(uuid.uuid4()),
                source="local",
                filename=original_name,
                transcript=transcript_result["text"],
                transcript_segments=transcript_result["segments"],
                ocr_text=ocr_text,
                created_at=datetime.now().isoformat(),
            )

            document = reel.model_dump()
            document["playlist_id"] = playlist_id
            document["stored_filename"] = stored_name

            data_path = DATA_DIR / f"{reel.id}.json"

            with open(data_path, "w", encoding="utf-8") as file:
                json.dump(
                    document,
                    file,
                    ensure_ascii=False,
                    indent=4,
                )

            for chunk_number, segment in enumerate(
                transcript_result["segments"]
            ):
                text = segment["text"].strip()

                if not text:
                    continue

                add_document(
                    text,
                    {
                        "id": reel.id,
                        "filename": reel.filename,
                        "source": reel.source,
                        "source_url": reel.source_url,
                        "playlist_id": playlist_id,
                        "chunk_number": chunk_number,
                        "start": segment["start"],
                        "end": segment["end"],
                    },
                )

            results.append(
                {
                    "id": reel.id,
                    "filename": original_name,
                    "playlist_id": playlist_id,
                    "status": "success",
                }
            )

        except Exception as error:
            print(f"Failed to process {original_name}: {error}")

            results.append(
                {
                    "filename": original_name,
                    "status": "failed",
                    "error": str(error),
                }
            )

        finally:
            await uploaded_file.close()

    # Update the local collection's video count.
    playlists = load_playlists()

    for playlist in playlists:
        if playlist["id"] == playlist_id:
            playlist["video_count"] = (
                playlist.get("video_count", 0)
                + sum(r["status"] == "success" for r in results)
            )
            break

    save_playlists(playlists)

    return {
        "playlist_id": playlist_id,
        "total": len(results),
        "succeeded": sum(r["status"] == "success" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "files": results,
    }


@app.get("/videos/{reel_id}")
def open_video(reel_id: str):
    data_path = DATA_DIR / f"{reel_id}.json"

    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    with open(data_path, "r", encoding="utf-8") as file:
        document = json.load(file)

    stored_name = document.get("stored_filename")

    if not stored_name:
        raise HTTPException(
            status_code=404,
            detail="Video file path is unavailable",
        )

    file_path = UPLOAD_DIR / Path(stored_name).name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video file not found",
        )

    return FileResponse(file_path)


@app.get("/ask")
def ask_question(query: str, playlist_id: str = "default"):
    if not get_playlist(playlist_id):
        raise HTTPException(
            status_code=404,
            detail="Playlist not found",
        )

    return answer_question(query, playlist_id=playlist_id)


# Preserve the original single-file upload endpoint.
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return await upload_files("default", [file])


@app.get("/youtube/connect")
def connect_youtube():
    if not GOOGLE_CLIENT_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing backend/client_secret.json. "
                "Configure Google OAuth first."
            ),
        )

    try:
        flow = Flow.from_client_secrets_file(
            str(GOOGLE_CLIENT_FILE),
            scopes=YOUTUBE_SCOPES,
            redirect_uri=YOUTUBE_REDIRECT_URI,
        )

        # Generate a PKCE verifier and challenge.
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            code_challenge_method="S256",
        )

        # Preserve the verifier generated for this exact OAuth request.
        if not flow.code_verifier:
            raise RuntimeError("OAuth did not generate a PKCE verifier.")

        pending_oauth_states[state] = flow.code_verifier

        return RedirectResponse(authorization_url)

    except HTTPException:
        raise
    except Exception as error:
        print(f"Could not start YouTube OAuth: {error}")
        raise HTTPException(
            status_code=500,
            detail="Could not start YouTube authorization. Check the backend terminal.",
        )


@app.get("/youtube/oauth/callback")
def youtube_oauth_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
):
    if error:
        print(f"YouTube authorization was declined or failed: {error}")
        raise HTTPException(
            status_code=400,
            detail="Google authorization was not completed. Please connect again.",
        )

    if not code or not state or state not in pending_oauth_states:
        print("OAuth state validation failed.")
        print(f"Received state: {state!r}")
        print(f"Pending state keys: {list(pending_oauth_states.keys())}")

        raise HTTPException(
            status_code=400,
            detail=(
                "YouTube authorization could not be verified. "
                "Check the backend terminal."
            ),
        )

    # Retrieve and remove the verifier for this OAuth request.
    code_verifier = pending_oauth_states.pop(state)

    try:
        flow = Flow.from_client_secrets_file(
            str(GOOGLE_CLIENT_FILE),
            scopes=YOUTUBE_SCOPES,
            state=state,
            redirect_uri=YOUTUBE_REDIRECT_URI,
        )

        # Critical fix: restore the verifier before exchanging the code.
        flow.code_verifier = code_verifier

        flow.fetch_token(code=code)

        credentials = flow.credentials

        YOUTUBE_TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

        print("YouTube OAuth completed successfully.")

    except Exception as error:
        print(f"YouTube OAuth failed: {error}")
        raise HTTPException(
            status_code=400,
            detail=(
                "YouTube authorization failed. "
                "Check the backend terminal for the underlying error."
            ),
        )

    return RedirectResponse("/app?youtube_connected=1")


@app.post("/youtube/import")
def import_youtube_playlist(payload: YouTubeImportRequest):
    service = get_youtube_service()

    if not service:
        raise HTTPException(
            status_code=401,
            detail=(
                "Connect your YouTube account first. "
                "If this is a private playlist, connect the Google account "
                "that has permission to access it."
            ),
        )

    name = payload.name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Collection name cannot be empty.",
        )

    playlist_id = extract_playlist_id(payload.playlist_url)
    playlists = load_playlists()

    if any(p["name"].casefold() == name.casefold() for p in playlists):
        raise HTTPException(
            status_code=409,
            detail="A collection with that name already exists.",
        )

    try:
        playlist_response = service.playlists().list(
            part="snippet,status",
            id=playlist_id,
            maxResults=1,
        ).execute()

        playlist_items = playlist_response.get("items", [])

        if not playlist_items:
            raise HTTPException(
                status_code=404,
                detail=(
                    "The playlist was not found or this account "
                    "cannot access it."
                ),
            )

        youtube_playlist = playlist_items[0]
        playlist_title = youtube_playlist["snippet"]["title"]

    except HTTPException:
        raise
    except Exception as error:
        print(f"Could not retrieve YouTube playlist: {error}")
        raise HTTPException(
            status_code=502,
            detail=(
                "YouTube could not return this playlist. "
                "Check the URL, account access, and API configuration."
            ),
        )

    collection = {
        "id": str(uuid.uuid4()),
        "name": name,
        "source": "youtube",
        "youtube_playlist_id": playlist_id,
        "created_at": datetime.now().isoformat(),
        "video_count": 0,
    }

    playlists.append(collection)
    save_playlists(playlists)

    results = []
    page_token = None

    try:
        while True:
            response = service.playlistItems().list(
                part="snippet,contentDetails,status",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token,
            ).execute()

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                details = item.get("contentDetails", {})

                video_id = (
                    details.get("videoId")
                    or snippet.get("resourceId", {}).get("videoId")
                )

                title = snippet.get("title") or "Untitled YouTube video"
                description = snippet.get("description") or ""

                if not video_id:
                    results.append(
                        {
                            "title": title,
                            "status": "failed",
                            "reason": "YouTube did not provide a video ID.",
                        }
                    )
                    continue

                source_url = (
                    f"https://www.youtube.com/watch?v={video_id}"
                )

                try:
                    transcript, segments = fetch_youtube_transcript(video_id)

                    searchable_text = transcript.strip() or (
                        f"Video title: {title}\n"
                        f"Video description: {description}"
                    ).strip()

                    reel = ReelDocument(
                        id=str(uuid.uuid4()),
                        source="youtube",
                        source_url=source_url,
                        filename=title,
                        caption=description,
                        transcript=transcript,
                        transcript_segments=segments,
                        created_at=datetime.now().isoformat(),
                    )

                    document = reel.model_dump()
                    document["playlist_id"] = collection["id"]
                    document["youtube_video_id"] = video_id
                    document["processing_status"] = (
                        "transcript_indexed"
                        if transcript
                        else "metadata_only"
                    )

                    data_path = DATA_DIR / f"{reel.id}.json"

                    with open(data_path, "w", encoding="utf-8") as file:
                        json.dump(
                            document,
                            file,
                            ensure_ascii=False,
                            indent=4,
                        )

                    if segments:
                        for chunk_number, segment in enumerate(segments):
                            text = segment["text"].strip()

                            if not text:
                                continue

                            add_document(
                                text,
                                {
                                    "id": reel.id,
                                    "filename": title,
                                    "source": "youtube",
                                    "source_url": source_url,
                                    "playlist_id": collection["id"],
                                    "youtube_video_id": video_id,
                                    "chunk_number": chunk_number,
                                    "start": segment["start"],
                                    "end": segment["end"],
                                },
                            )
                    else:
                        add_document(
                            searchable_text,
                            {
                                "id": reel.id,
                                "filename": title,
                                "source": "youtube",
                                "source_url": source_url,
                                "playlist_id": collection["id"],
                                "youtube_video_id": video_id,
                                "chunk_number": 0,
                            },
                        )

                    results.append(
                        {
                            "id": reel.id,
                            "title": title,
                            "source_url": source_url,
                            "status": (
                                "transcript_indexed"
                                if transcript
                                else "metadata_only"
                            ),
                        }
                    )

                except Exception as error:
                    print(
                        f"Failed to import YouTube video {video_id}: {error}"
                    )
                    results.append(
                        {
                            "title": title,
                            "source_url": source_url,
                            "status": "failed",
                            "reason": str(error),
                        }
                    )

            page_token = response.get("nextPageToken")

            if not page_token:
                break

    except Exception as error:
        print(f"YouTube playlist import failed: {error}")
        raise HTTPException(
            status_code=502,
            detail=(
                "An error occurred while reading playlist items from YouTube."
            ),
        )

    succeeded = sum(
        result["status"] in {"transcript_indexed", "metadata_only"}
        for result in results
    )

    transcript_count = sum(
        result["status"] == "transcript_indexed"
        for result in results
    )

    metadata_count = sum(
        result["status"] == "metadata_only"
        for result in results
    )

    failed_count = sum(
        result["status"] == "failed"
        for result in results
    )

    # Update the collection's imported video count.
    playlists = load_playlists()

    for playlist in playlists:
        if playlist["id"] == collection["id"]:
            playlist["video_count"] = succeeded
            break

    save_playlists(playlists)

    return {
        "collection_id": collection["id"],
        "collection_name": name,
        "youtube_playlist_title": playlist_title,
        "total": len(results),
        "imported": succeeded,
        "transcripts_indexed": transcript_count,
        "metadata_only": metadata_count,
        "failed": failed_count,
        "videos": results,
    }
