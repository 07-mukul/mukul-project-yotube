from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.api_core import exceptions as google_api_exceptions
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, InvalidVideoId
from youtube_transcript_api._errors import IpBlocked, TranscriptsDisabled
from flask_cors import CORS
import os
import re
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import time
import requests
import http.cookiejar

try:
    import grpc
except ImportError:
    grpc = None

_BASE_DIR = Path(__file__).resolve().parent
# override=True: .env wins over a stale GEMINI_API_KEY from Windows system env (common "it worked then broke" cause)
load_dotenv(_BASE_DIR / ".env", override=True)


def _reload_env_from_file() -> None:
    """Re-read .env so key/model changes apply without restarting Flask."""
    load_dotenv(_BASE_DIR / ".env", override=True)


def _read_gemini_api_key() -> str:
    """Read key from env; strip BOM, quotes, and whitespace (common .env mistakes)."""
    raw = os.getenv("GEMINI_API_KEY") or ""
    return raw.strip().strip("\ufeff").strip('"').strip("'")


app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

def _get_genai_client():
    api_key = _read_gemini_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

try:
    client = _get_genai_client()
    if client:
        print(f"[env] GEMINI_API_KEY loaded and GenAI client initialized")
    else:
        print("[env] WARNING: GEMINI_API_KEY is not set. The app will run but summaries will fail.")
except Exception as e:
    print(f"[env] CRITICAL: Failed to initialize GenAI client. Error: {e}")

# Simple cache for summaries: {video_id: {"summary": ..., "language": ..., timestamp: ...}}
summary_cache = {}
CACHE_DURATION = 86400  # Cache for 24 hours to avoid YouTube rate limiting

# Request throttling - track last request time
last_request_time = {}
MIN_REQUEST_INTERVAL = 5  # Minimum 5 seconds between requests for same video


def _looks_like_invalid_gemini_key(error_msg: str) -> bool:
    m = error_msg.lower()
    return any(
        phrase in m
        for phrase in (
            "api_key_invalid",
            "api key not found",
            "pass a valid api key",
            "invalid api key",
        )
    )


def _looks_like_gemini_rate_limit(error_msg: str) -> bool:
    """Match HTTP handler to Gemini exhaustion messages (not only literal '429')."""
    m = error_msg.lower()
    return any(
        phrase in m
        for phrase in (
            "429",
            "quota",
            "rate limit",
            "rate_limit",
            "rate-limited",
            "resource exhausted",
            "resource_exhausted",
            "hit rate limits",
            "still rate",
            "all api key",
            "unique key",
            "try again in a few moments",
        )
    )


@app.route('/', methods=['GET'])
def home():
    response = send_from_directory('.', 'index.html')
    # Prevent caching of the main page during deployment updates
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "active",
        "message": "Service is running"
    }), 200


@app.route('/api-info', methods=['GET'])
def api_info():
    return jsonify({
        "status": "API is running",
        "endpoints": {
            "GET /": "Frontend application",
            "GET /health": "Health check",
            "GET /summary?url=YOUTUBE_URL": "Summarize YouTube video from URL"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "active", "message": "Service is running"}), 200


@app.after_request
def _no_cache_summary_responses(response):
    """Prevent browsers/CDNs from caching GET /summary (was returning video 1's body for other videos)."""
    if request.path == "/summary":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def extract_video_id(youtube_url):
    """Extract 11-character video ID from a YouTube URL or raw ID."""
    if not youtube_url:
        return None
    s = youtube_url.strip()

    if re.match(r"^[a-zA-Z0-9_-]{11}$", s):
        return s

    m = re.search(r"(?:youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})", s)
    if m:
        return m.group(1)

    m = re.search(r"youtube\.com/(?:shorts|embed|live)/([a-zA-Z0-9_-]{11})", s)
    if m:
        return m.group(1)

    m = re.search(r"youtube\.com/v/([a-zA-Z0-9_-]{11})(?:\?|$|/)", s)
    if m:
        return m.group(1)

    if "youtube.com" in s or "youtube-nocookie.com" in s or "music.youtube.com" in s:
        parsed = urlparse(s)
        for vid in parse_qs(parsed.query).get("v", []):
            if vid and len(vid) == 11:
                return vid

    return None

@app.route('/summary', methods=['GET'])
def youtube_summarizer():
    youtube_url = request.args.get('url', '').strip()
    demo_mode = request.args.get('demo', 'false').lower() == 'true'
    
    # Demo mode for testing without API calls
    if demo_mode:
        return jsonify({
            "data": """Here is a 10-point summary of the YouTube video based on its transcript:

1. The video begins with an engaging introduction and overview of the main topic
2. Key concepts are explained with clear examples and visual demonstrations
3. The presenter discusses the historical context and background information
4. Practical applications and real-world use cases are presented
5. Technical details are broken down into understandable segments
6. Common misconceptions and myths are addressed and clarified
7. Expert insights and tips for success are shared throughout
8. The content transitions smoothly between different interconnected topics
9. Concluding remarks summarize the main takeaways effectively
10. The video ends with a call-to-action or invitation for further engagement""",
            "error": False,
            "language": "English",
            "available_languages": ["en", "es", "fr"],
            "demo": True
        }), 200
    
    # Validate URL is provided
    if not youtube_url:
        return jsonify({
            "data": "Missing YouTube URL. Please provide 'url' query parameter (e.g., https://youtube.com/watch?v=... or https://youtu.be/...)",
            "error": True
        }), 400
    
    # Extract video ID from URL
    video_id = extract_video_id(youtube_url)
    
    if not video_id:
        return jsonify({
            "data": "Invalid YouTube URL or Video ID. Please provide a valid YouTube link or video ID (11 characters)",
            "error": True
        }), 400

    print(f"[summary] video_id={video_id}")
    
    # Check cache first (only for successful results)
    if video_id in summary_cache:
        cached_data = summary_cache[video_id]
        if time.time() - cached_data['timestamp'] < CACHE_DURATION:
            print(f"Cache hit for video {video_id}")
            return jsonify({
                "data": cached_data['summary'],
                "error": False,
                "video_id": video_id,
                "language": cached_data['language'],
                "available_languages": cached_data.get('available_languages', []),
                "cached": True
            }), 200
        else:
            # Cache expired, remove it
            del summary_cache[video_id]
    
    try:
        transcript_data = get_transcript(video_id)
        summary = generate_summary(transcript_data['text'], transcript_data['language'])
        
        # Cache the result
        summary_cache[video_id] = {
            'summary': summary,
            'language': transcript_data.get('language', 'Unknown'),
            'available_languages': transcript_data.get('available_languages', []),
            'timestamp': time.time()
        }
        
    except NoTranscriptFound:
        return jsonify({"data": "No Subtitles found. Try videos with English or Hindi subtitles.", "error": True}), 404
    except InvalidVideoId:
        return jsonify({"data": "Invalid Video Id", "error": True}), 400
    except Exception as e:
        print(f"Error: {e}")
        error_msg = str(e)
        
        # Handle specific YouTube errors
        if "blocking" in error_msg.lower() or "ip blocked" in error_msg.lower():
            return jsonify({
                "data": "YouTube is blocking your IP due to too many requests. Solutions: 1) Wait 15-30 minutes, 2) Switch VPN server, 3) Try a different video. Cached results will be available immediately.",
                "error": True,
                "error_type": "ip_blocking",
                "solutions": [
                    "Wait 15-30 minutes for YouTube to unblock your IP",
                    "Switch to a different VPN server",
                    "Try summarizing a different video",
                    "Ensure VPN is properly connected"
                ]
            }), 429
        elif "no transcripts" in error_msg.lower() or "transcripts disabled" in error_msg.lower():
            return jsonify({"data": "This video does not have subtitles available.", "error": True}), 404
        elif "No transcripts available" in error_msg:
            return jsonify({"data": "No Subtitles found. Try videos with English or Hindi subtitles.", "error": True}), 404
        elif _looks_like_invalid_gemini_key(error_msg):
            return jsonify({
                "data": (
                    "Gemini rejected your API key (invalid or revoked). "
                    "Create a new key at https://aistudio.google.com/apikey , put it in .env as GEMINI_API_KEY=..., "
                    "save the file, and restart the Flask server."
                ),
                "error": True,
                "error_type": "invalid_api_key",
            }), 401
        elif _looks_like_gemini_rate_limit(error_msg):
            return jsonify({
                "data": "API Rate limit exceeded. Please try again in a few moments.",
                "error": True,
                "error_type": "rate_limit"
            }), 429
        
        return jsonify({"data": f"Unable to Summarize the video: {error_msg}", "error": True}), 500

    return jsonify({
        "data": summary,
        "error": False,
        "video_id": video_id,
        "language": transcript_data.get('language', 'Unknown'),
        "available_languages": transcript_data.get('available_languages', []),
        "cached": False
    }), 200


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _load_cookies_into_session(session):
    """Load cookies from cookies.txt into the requests session if it exists."""
    cookies_path = os.path.join(os.path.dirname(__file__), 'cookies.txt')
    if os.path.exists(cookies_path):
        try:
            cj = http.cookiejar.MozillaCookieJar(cookies_path)
            cj.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(cj)
            print(f"[auth] Successfully loaded cookies from {cookies_path}")
            return True
        except Exception as e:
            print(f"[auth] Failed to load cookies: {e}")
    return False


def _direct_http_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    _load_cookies_into_session(s)
    return s


def _proxy_http_session():
    """Return a requests Session using Webshare, or None if env is incomplete."""
    host = os.getenv("WEBSHARE_PROXY_HOST")
    port = os.getenv("WEBSHARE_PROXY_PORT")
    user = os.getenv("WEBSHARE_PROXY_USERNAME")
    password = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if not all([host, port, user, password]):
        return None
    proxy_url = f"http://{user}:{password}@{host}:{port}"
    s = requests.Session()
    s.proxies.update({"http": proxy_url, "https": proxy_url})
    s.headers.update({"User-Agent": UA})
    _load_cookies_into_session(s)
    return s


def _fetch_transcript_with_session(video_id, session):
    """Core transcript fetch; raises library errors or Exception for no transcript text."""
    yt_api = YouTubeTranscriptApi(http_client=session)
    
    try:
        # The session already contains cookies if they were available
        transcript_list = yt_api.list(video_id)
    except Exception as e:
        print(f"Failed to list transcripts: {e}")
        raise

    available_langs = set()
    available_langs.update(transcript_list._generated_transcripts.keys())
    available_langs.update(transcript_list._manually_created_transcripts.keys())
    available_langs = list(available_langs)

    print(f"Available languages: {available_langs}")

    language_used = None
    transcript_response = None

    if any(lang.startswith("en") for lang in available_langs):
        try:
            transcript_response = yt_api.fetch(video_id, languages=["en"])
            language_used = "English"
            print("[ok] Using English transcript")
        except Exception as e:
            print(f"Failed to fetch English: {e}")

    if transcript_response is None and any(lang.startswith("hi") for lang in available_langs):
        try:
            transcript_response = yt_api.fetch(video_id, languages=["hi"])
            language_used = "Hindi"
            print("[ok] Using Hindi transcript")
        except Exception as e:
            print(f"Failed to fetch Hindi: {e}")

    if transcript_response is None and available_langs:
        try:
            first_lang = available_langs[0]
            transcript_response = yt_api.fetch(video_id, languages=[first_lang])
            language_used = f"Language: {first_lang}"
            print(f"[ok] Using {first_lang} transcript")
        except Exception as e:
            print(f"Failed to fetch {first_lang}: {e}")

    if transcript_response is None:
        raise Exception(f"No transcripts available. Available languages: {available_langs}")

    transcript_text = " ".join([snippet.text for snippet in transcript_response])
    print(f"[ok] Successfully fetched transcript with {len(transcript_text)} characters")

    return {
        "text": transcript_text,
        "language": language_used,
        "available_languages": available_langs,
    }


def _get_transcript_one_route(video_id, label, session, max_retries=2, base_delay=1):
    """Try transcript fetch with retries on transient failures (e.g. IpBlocked)."""
    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"[{label}] attempt {attempt + 1}/{max_retries}")
            return _fetch_transcript_with_session(video_id, session)
        except InvalidVideoId:
            raise
        except TranscriptsDisabled as e:
            print(f"Transcripts disabled: {e}")
            raise Exception("This video does not have transcripts available.") from e
        except IpBlocked as e:
            last_error = e
            print(f"[{label}] IP blocked by YouTube: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                break
        except Exception as e:
            last_error = e
            print(f"[{label}] Transcript fetch failed: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                break
    if last_error is not None:
        raise last_error
    raise Exception("Transcript fetch failed")


def get_transcript(video_id):
    """Fetch transcript: try direct connection first, then Webshare proxy if configured."""
    routes = [("direct", _direct_http_session())]
    proxy_sess = _proxy_http_session()
    if proxy_sess:
        ph = os.getenv("WEBSHARE_PROXY_HOST")
        pp = os.getenv("WEBSHARE_PROXY_PORT")
        routes.append((f"proxy {ph}:{pp}", proxy_sess))

    last_exc = None
    for label, session in routes:
        try:
            return _get_transcript_one_route(video_id, label, session)
        except InvalidVideoId:
            raise
        except IpBlocked as e:
            last_exc = e
            print(f"Route {label} failed with IpBlocked, trying next route if any...")
            continue
        except Exception as e:
            err = str(e).lower()
            if "blocking" in err or "ip blocked" in err or "429" in err:
                last_exc = e
                print(f"Route {label} failed ({e}), trying next route if any...")
                continue
            raise

    if last_exc is not None:
        raise Exception(
            "YouTube is blocking transcript requests from this network. "
            "Try: wait and retry, use a VPN/residential proxy (set Webshare env vars), or a different network."
        ) from last_exc
    raise Exception("Unable to fetch transcript.")

def _collect_exception_chain(exc: BaseException | None) -> list[BaseException]:
    """Collect nested exceptions (__cause__, __context__); SDK often wraps gRPC / API errors."""
    out: list[BaseException] = []
    seen: set[int] = set()

    def walk(e: BaseException | None) -> None:
        if e is None or id(e) in seen:
            return
        seen.add(id(e))
        out.append(e)
        walk(e.__cause__)
        if e.__context__ is not e.__cause__:
            walk(e.__context__)

    walk(exc)
    return out


def _gemini_error_text(exc: BaseException) -> str:
    parts = []
    for e in _collect_exception_chain(exc):
        parts.append(str(e))
        if getattr(e, "args", None):
            parts.extend(str(a) for a in e.args if a)
    return " ".join(parts).lower()


def _is_retryable_gemini_quota_or_rate(exc: BaseException) -> bool:
    """Transient quota / overload / RPC — retry with backoff or next key."""
    for e in _collect_exception_chain(exc):
        if isinstance(
            e,
            (
                google_api_exceptions.ResourceExhausted,
                google_api_exceptions.TooManyRequests,
                google_api_exceptions.ServiceUnavailable,
                google_api_exceptions.DeadlineExceeded,
                google_api_exceptions.InternalServerError,
            ),
        ):
            return True
        if grpc and isinstance(e, grpc.RpcError):
            code = e.code()
            if code in (
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.ABORTED,
            ):
                return True
    text = _gemini_error_text(exc)
    return any(
        phrase in text
        for phrase in (
            "429",
            "quota",
            "rate limit",
            "rate_limit",
            "resource exhausted",
            "resource_exhausted",
            "too many requests",
            "exceeded your",
            "exhausted",
            "throttl",
            "capacity",
            "try again later",
            "unavailable",
            "temporarily",
            "slow down",
            "overloaded",
        )
    )


GEMINI_ATTEMPTS = 3
GEMINI_BACKOFF_BASE_SEC = 2.0


def generate_summary(transcript, language="English"):
    _reload_env_from_file()
    api_key = _read_gemini_api_key()
    if not api_key:
        raise Exception(
            "No GEMINI_API_KEY in .env. Add GEMINI_API_KEY=your_key to the .env file next to app.py."
        )

    # Re-initialize client if necessary or use the global one
    current_client = genai.Client(api_key=api_key)
    model_name = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip() or "gemini-2.0-flash"

    if "Hindi" in language:
        prompt = (
            "You are a friendly and helpful assistant. Summarize the following YouTube video transcript "
            "in Hindi. Provide exactly 10 clear, engaging points. Make the tone friendly and easy to understand. "
            f"Transcript: {transcript}"
        )
    else:
        prompt = (
            "You are a friendly and helpful assistant. Summarize the following YouTube video transcript "
            "in English. Provide exactly 10 clear, engaging points. Make the tone friendly and easy to understand. "
            f"Transcript: {transcript}"
        )

    last_error: Exception | None = None
    for attempt in range(GEMINI_ATTEMPTS):
        try:
            print(f"[key] Gemini attempt {attempt + 1}/{GEMINI_ATTEMPTS} model={model_name}")
            response = current_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            print("[ok] Summary generated successfully")
            return response.text

        except Exception as e:
            last_error = e
            detail = _gemini_error_text(e)
            print(f"[fail] {type(e).__name__}: {detail[:450]}")

            if not _is_retryable_gemini_quota_or_rate(e):
                raise Exception(f"Unable to generate summary: {detail}") from e

            if attempt + 1 < GEMINI_ATTEMPTS:
                delay = GEMINI_BACKOFF_BASE_SEC * (2**attempt)
                print(f"[warn] Transient limit; sleeping {delay:.1f}s then retry...")
                time.sleep(delay)

    msg = (
        f"Unable to generate summary: still rate-limited after {GEMINI_ATTEMPTS} attempt(s) with backoff. "
        "Wait 1–2 minutes, try GEMINI_MODEL=gemini-1.5-flash, or enable billing for higher quota."
    )
    if last_error:
        raise Exception(msg) from last_error
    raise Exception(msg)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
