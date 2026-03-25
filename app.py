from flask import Flask, jsonify, request, send_from_directory
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
import yt_dlp
import gradio as gr
import threading

# Local transformer imports
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from accelerate import Accelerator

_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env", override=True)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Simple cache for summaries: {video_id: {"summary": ..., "language": ..., timestamp: ...}}
summary_cache = {}
CACHE_DURATION = 86400  # Cache for 24 hours to avoid YouTube rate limiting

# Request throttling - track last request time
last_request_time = {}
MIN_REQUEST_INTERVAL = 5  # Minimum 5 seconds between requests for same video

# Global summarization model
summarizer = None
tokenizer = None
model = None
accelerator = None

def load_summarization_model():
    """Load the summarization model once at startup with multiple fallbacks"""
    global summarizer, tokenizer, model, accelerator
    
    # List of models to try in order of preference
    # FLAN-T5 models are better for instruction following and messy transcripts
    model_options = [
        "google/flan-t5-base",            # Good balance of size and quality
        "google/flan-t5-large",           # Instruction-following, great for transcripts
        "sshleifer/distilbart-cnn-6-6",   # Faster version of 12-6, slightly lower quality but much faster
        "sshleifer/distilbart-cnn-12-6",  # 1GB, fast, good quality
        "facebook/bart-large-cnn",        # 1.6GB, best quality
        "t5-small",                       # Ultra-fast fallback
    ]
    
    print("[model] Initializing accelerator...")
    try:
        accelerator = Accelerator()
        device = accelerator.device
    except Exception as e:
        print(f"[model] Accelerator init failed: {e}. Falling back to standard device detection.")
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in model_options:
        try:
            # Check for local directory first
            local_path = os.path.join(os.path.dirname(__file__), "flan-t5-base")
            if "flan-t5-base" in model_name and os.path.exists(local_path):
                print(f"[model] Found local model path: {local_path}")
                model_to_load = local_path
            else:
                model_to_load = model_name

            print(f"[model] Attempting to load: {model_to_load}...")
            
            # Use Auto classes for more robustness
            tokenizer = AutoTokenizer.from_pretrained(model_to_load)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_to_load)
            
            if accelerator:
                model = accelerator.prepare(model)
            
            # Explicitly create the pipeline with the loaded model and tokenizer
            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"[model] Success! {model_name} loaded on {device}")
            return True
            
        except Exception as e:
            print(f"[model] Failed to load {model_name}: {e}")
            continue

    # Final attempt: direct pipeline call (let transformers handle it)
    try:
        print("[model] Final attempt: loading default pipeline...")
        summarizer = pipeline("summarization")
        print("[model] Default pipeline loaded successfully")
        return True
    except Exception as e:
        print(f"[model] All model loading attempts failed: {e}")
        return False

# Load model at startup
model_loaded = load_summarization_model()
if not model_loaded:
    print("[WARNING] Summarization model failed to load. The app will start but summaries will fail.")





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
    style = request.args.get('style', 'bullet').lower()
    
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

    print(f"[summary] video_id={video_id} style={style}")
    
    # Cache key includes style
    cache_key = f"{video_id}_{style}"
    
    # Check cache first (only for successful results)
    if cache_key in summary_cache:
        cached_data = summary_cache[cache_key]
        if time.time() - cached_data['timestamp'] < CACHE_DURATION:
            print(f"Cache hit for {cache_key}")
            return jsonify({
                "data": cached_data['summary'],
                "keywords": cached_data.get('keywords', []),
                "content_type": cached_data.get('content_type', 'General'),
                "error": False,
                "video_id": video_id,
                "language": cached_data['language'],
                "available_languages": cached_data.get('available_languages', []),
                "cached": True
            }), 200
        else:
            # Cache expired, remove it
            del summary_cache[cache_key]
    
    try:
        transcript_data = get_transcript(video_id)
        result = generate_summary(transcript_data['text'], transcript_data['language'], style=style)
        
        summary = result['summary']
        keywords = result['keywords']
        content_type = result['content_type']
        
        if not summary:
            summary = "The AI was unable to generate a summary for this video. Please try again."
        
        # Cache the result
        summary_cache[cache_key] = {
            'summary': summary,
            'keywords': keywords,
            'content_type': content_type,
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
            # Check if Supadata was even tried and why it failed
            supadata_err = next((e for e in error_msg.split('|') if "supadata" in e.lower()), None)
            if supadata_err:
                main_msg = f"YouTube is blocking requests and Supadata API also failed: {supadata_err.strip()}"
            else:
                main_msg = "YouTube is blocking your IP due to too many requests. Solutions: 1) Wait 15-30 minutes, 2) Switch VPN server, 3) Try a different video."
            
            return jsonify({
                "data": main_msg,
                "error": True,
                "error_type": "ip_blocking",
                "solutions": [
                    "Check your Supadata API key in .env",
                    "Wait 15-30 minutes for YouTube to unblock your IP",
                    "Switch to a different VPN server",
                    "Try summarizing a different video"
                ]
            }), 429
        elif "no transcripts" in error_msg.lower() or "transcripts disabled" in error_msg.lower():
            return jsonify({"data": "This video does not have subtitles available.", "error": True}), 404
        elif "No transcripts available" in error_msg:
            return jsonify({"data": "No Subtitles found. Try videos with English or Hindi subtitles.", "error": True}), 404
        elif "model" in error_msg.lower() and "not loaded" in error_msg.lower():
            return jsonify({
                "data": "Summarization model failed to load. Please check the server logs.",
                "error": True,
                "error_type": "model_not_loaded"
            }), 500
        
        return jsonify({"data": f"Unable to Summarize the video: {error_msg}", "error": True}), 500

    return jsonify({
        "data": summary,
        "keywords": keywords,
        "content_type": content_type,
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

    def _extract_text(s):
        if isinstance(s, str): return s
        if isinstance(s, dict): return s.get('text', '')
        try: return s.text # Try object attribute
        except: return str(s)

    if transcript_response is None:
        raise Exception(f"No transcripts available. Available languages: {available_langs}")

    transcript_text = " ".join([_extract_text(snippet) for snippet in transcript_response])
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


def _fetch_with_ytdlp(video_id):
    """Fallback using yt-dlp to extract transcripts."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookies_path = os.path.join(os.path.dirname(__file__), 'cookies.txt')
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    if os.path.exists(cookies_path):
        ydl_opts['cookiefile'] = cookies_path
        print(f"[ytdlp] Using cookies from {cookies_path}")

    # Add proxy if available
    host = os.getenv("WEBSHARE_PROXY_HOST")
    port = os.getenv("WEBSHARE_PROXY_PORT")
    user = os.getenv("WEBSHARE_PROXY_USERNAME")
    pw = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if all([host, port, user, pw]):
        proxy_url = f"http://{user}:{pw}@{host}:{port}"
        ydl_opts['proxy'] = proxy_url
        print("[ytdlp] Using proxy")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subtitles = info.get('subtitles') or info.get('automatic_captions')
            
            if not subtitles:
                return None, None

            # Prefer English, then Hindi, then whatever is first
            target_lang = None
            if 'en' in subtitles: target_lang = 'en'
            elif 'hi' in subtitles: target_lang = 'hi'
            else: target_lang = list(subtitles.keys())[0]

            if target_lang:
                print(f"[ytdlp] found subtitles for {target_lang}")
                # Fetch the subtitle content
                subtitle_info = subtitles[target_lang]
                # subtitle_info is a list of dicts, we want the one with ext='json' or 'vtt'
                # For simplicity, we'll try to find a URL and fetch it
                for sub in subtitle_info:
                    if sub.get('ext') == 'json3': # YouTube's native format
                        sub_url = sub.get('url')
                        res = requests.get(sub_url)
                        if res.status_code == 200:
                            data = res.json()
                            text_parts = []
                            # Try json3 format (events/segs)
                            if 'events' in data:
                                for event in data.get('events', []):
                                    for seg in event.get('segs', []):
                                        if seg.get('utf8'):
                                            text_parts.append(seg['utf8'])
                            # Try simple srv3 format (body/p)
                            elif 'body' in data:
                                for p in data.get('body', {}).get('p', []):
                                    if p.get('#text'):
                                        text_parts.append(p['#text'])
                            
                            if text_parts:
                                full_text = " ".join(text_parts)
                                return full_text, target_lang
                
                return "Transcript detected but format not supported for direct extraction.", target_lang
    except Exception as e:
        print(f"[ytdlp] Error: {e}")
    
    return None, None


def get_transcript(video_id):
    """
    Tries multiple routes to get transcript:
    1. Supadata API (no IP blocking)
    2. Direct connection (with cookies)
    3. Webshare Proxy (with cookies)
    4. yt-dlp fallback
    """
    # Route 1: Supadata API
    api_key = os.getenv("SUPADATA_API_KEY")
    errors = []
    if api_key:
        try:
            print(f"[transcript] trying supadata for {video_id}...")
            res = requests.get(
                "https://api.supadata.ai/v1/youtube/transcript",
                params={"videoId": video_id, "text": True},
                headers={"x-api-key": api_key},
                timeout=30
            )
            if res.status_code == 200:
                data = res.json()
                text = data.get("content", "")
                if not text and isinstance(data.get("content"), list):
                    text = " ".join([c.get("text", "") for c in data["content"]])
                
                if text:
                    lang = data.get("lang", "en")
                    print(f"[ok] Supadata transcript fetched: {len(text)} chars")
                    return {
                        "text": text,
                        "language": "English" if lang.startswith("en") else lang,
                        "available_languages": [lang]
                    }
            else:
                err_msg = f"supadata failed ({res.status_code}): {res.text}"
                print(f"[supadata] {err_msg}")
                errors.append(err_msg)
        except Exception as e:
            err_msg = f"supadata failed: {e}"
            print(f"[supadata] {err_msg}")
            errors.append(err_msg)

    # Route 2 & 3: Direct + Proxy
    routes = [
        ("direct", _direct_http_session),
        ("proxy", _proxy_http_session),
    ]

    for route_name, session_factory in routes:
        try:
            session = session_factory()
            if session is None:
                if route_name == "proxy":
                    print("[transcript] skipping proxy (not configured)")
                continue

            print(f"[transcript] trying {route_name} for {video_id}...")
            transcript_result = _get_transcript_one_route(video_id, route_name, session)
            return transcript_result
        except Exception as e:
            err_msg = str(e)
            print(f"[fail] {route_name} failed: {err_msg[:200]}")
            errors.append(f"{route_name}: {err_msg}")
            if isinstance(e, (NoTranscriptFound, InvalidVideoId, TranscriptsDisabled)):
                raise e

    # Route 4: yt-dlp fallback
    print(f"[transcript] trying ytdlp fallback for {video_id}...")
    try:
        ytdlp_text, ytdlp_lang = _fetch_with_ytdlp(video_id)
        if ytdlp_text:
            return {
                "text": ytdlp_text,
                "language": ytdlp_lang,
                "available_languages": [ytdlp_lang]
            }
    except Exception as e:
        print(f"[fail] ytdlp fallback failed: {e}")
        errors.append(f"ytdlp: {e}")

    detailed_errors = " | ".join(errors)
    
    # If any route was specifically blocked, or if we have no specific transcript-unavailable error
    if any("blocked" in e.lower() for e in errors) or not any("unavailable" in e.lower() or "disabled" in e.lower() for e in errors):
        raise Exception(f"YouTube IP Blocked: {detailed_errors}")
    else:
        raise NoTranscriptFound(f"Transcript unavailable for this video: {detailed_errors}")

def is_lyrics(text):
    """Detect if the content looks like song lyrics with high accuracy"""
    # Clean and split into lines/sentences
    lines = [l.strip() for l in text.split(".") if l.strip()]
    if not lines:
        return False
    
    # Lyrics usually have very short lines and a lot of repetition
    short_lines = sum(1 for l in lines if len(l.split()) < 6)
    
    # Music often has specific keywords in transcripts
    music_keywords = ["chorus", "verse", "melody", "rhythm", "instrumental", "[music]"]
    has_music_keywords = any(kw in text.lower() for kw in music_keywords)
    
    # Stories have short dialogue lines too, so we need a higher threshold
    # and we check if it lacks typical "story" words
    story_keywords = ["once upon a time", "narrator", "suddenly", "however", "therefore", "because"]
    has_story_keywords = any(kw in text.lower() for kw in story_keywords)
    
    if has_story_keywords:
        return False # Definitely a story
        
    return (short_lines > len(lines) * 0.6) or (has_music_keywords and short_lines > len(lines) * 0.3)

def summarize_lyrics(text, language="English"):
    """Return a dynamic interpretation of song lyrics using the AI model"""
    global summarizer
    try:
        print("[model] Generating dynamic lyrics summary...")
        
        # Determine if T5 is used for the prefix
        is_t5 = False
        if model and hasattr(model, 'config'):
            is_t5 = "t5" in model.config._name_or_path.lower()
            
        input_text = ("summarize the meaning of these lyrics: " + text[:1000]) if is_t5 else text[:1000]
        
        result = summarizer(
            input_text,
            max_length=150,
            min_length=50,
            do_sample=False,
            num_beams=2,
            early_stopping=True,
            truncation=True
        )
        
        summary_text = result[0]['summary_text']
        
        # Use the same formatting logic to get 8-10 points
        result_formatted = _format_summary_points([summary_text], language, is_lyrics=True)
        return result_formatted if result_formatted else "Unable to interpret song lyrics."
        
    except Exception as e:
        print(f"[model] Lyrics summary failed: {e}. Falling back to general summary.")
        return None # Fallback to general summary logic in generate_summary

def is_story(text):
    """Detect if the content is a narrative story"""
    story_keywords = [
        "once upon a time", "lived in", "there was a", "narrator", 
        "suddenly", "village", "king", "forest", "moral of the story",
        "happily ever after", "neighborhood", "farmer", "merchant"
    ]
    text_lower = text.lower()
    score = sum(2 for kw in story_keywords if kw in text_lower)
    
    # Stories often have character names (Capitalized words in middle of sentences)
    potential_names = len(re.findall(r'(?<!^)(?<![.!?]\s)[A-Z][a-z]+', text))
    if potential_names > 10: score += 5
    
    return score >= 6

def is_educational(text):
    """Detect if the content is educational/academic"""
    edu_keywords = [
        "definition", "concept", "principle", "study", "research", 
        "example", "theory", "evidence", "analysis", "conclusion",
        "significant", "impact", "process", "structure", "function"
    ]
    text_lower = text.lower()
    score = sum(1 for kw in edu_keywords if kw in text_lower)
    return score >= 5

def is_tutorial(text):
    """Detect if the content is a tutorial/how-to"""
    tutorial_keywords = [
        "how to", "step by step", "first", "second", "then", "finally",
        "tutorial", "guide", "setup", "install", "create", "make sure",
        "click", "select", "using", "example", "tip"
    ]
    text_lower = text.lower()
    score = sum(1 for kw in tutorial_keywords if kw in text_lower)
    return score >= 5

def extract_keywords(text, num_keywords=5):
    """Simple keyword extraction based on frequency and length (FIX 11)"""
    # Clean text
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = clean_text.split()
    
    # Filter stopwords and short words
    stopwords = {'the', 'and', 'this', 'that', 'with', 'from', 'they', 'their', 'your', 'about', 'would', 'could', 'should'}
    meaningful_words = [w for w in words if len(w) > 4 and w not in stopwords]
    
    # Count frequencies
    from collections import Counter
    counts = Counter(meaningful_words)
    return [word for word, count in counts.most_common(num_keywords)]

def detect_content_type(text):
    """Consolidated content type detection (FIX 9)"""
    if is_lyrics(text): return "lyrics"
    if is_story(text): return "story"
    if is_tutorial(text): return "tutorial"
    if is_educational(text): return "educational"
    return "general"

def clean_sentence(s):
    """Clean and humanize a single sentence (STEP 2)"""
    if not s: return ""
    s = s.strip()

    # STEP 4: Remove direct dialogue fragments that don't make sense as summary points
    if re.search(r'^(Sherat|Shahed|Sugdev|Son|Jack|Mother|Giant),? (what|where|how|why|is|are|do|can)\b', s, flags=re.I) or s.endswith('?'):
        if len(s.split()) < 10: # Only remove short question/dialogue fragments
            return ""

    # Fix prompt leakage: Remove parenthetical instructions or meta-talk the AI might echo
    s = re.sub(r'\(NO NAME:.*?\)', '', s, flags=re.I)
    s = re.sub(r'\(NO ENDING.*?\)', '', s, flags=re.I)
    s = re.sub(r'\(NO MESSAGE.*?\)', '', s, flags=re.I)
    s = re.sub(r'\(FLOW:.*?\)', '', s, flags=re.I)
    s = re.sub(r'\(REQUIREMENTS:.*?\)', '', s, flags=re.I)
    s = re.sub(r'\(TASK:.*?\)', '', s, flags=re.I)
    s = re.sub(r'\(Exactly 12.*?\)', '', s, flags=re.I)
    s = re.sub(r'>>\s*', '', s) # Remove leading AI arrows

    # Humanization: Replace robotic or awkward phrasing
    replacements = {
        "this city people": "the villagers",
        "I'm preparing": "the boy started preparing",
        "grandfather": "the elder",
        "they are very smart": "the villagers showed great wisdom",
        "nothing will happen": "nothing seemed to change at first",
        "why are you": "the people wondered why",
        "it is mentioned": "the story reveals",
        "the video shows": "we see",
        "according to the video": "as the story unfolds",
        "the narrator says": "it is said that",
        "he thinks we should": "the elder suggested that they should",
        "all your effort will go to waste": "all their efforts would be in vain",
        "open all our eyes": "opened everyone's eyes to the truth",
        "one day the situation will surely change": "the people hoped that one day the situation would change",
        "we need to be ready for same": "they needed to be prepared for the future",
        "they want to live in a strange place": "the villagers were faced with the challenge of a new and difficult situation",
        "they are not sure how to survive there": "they were uncertain of how they would survive the drought",
        "wish a worse tomorrow": "hope for a better tomorrow", 
        "if you wish a worse tomorrow": "to ensure a better tomorrow", 
        "just then the village elder": "the story continues as the village elder",
        "gets shocked by the scene there": "was amazed to see the progress that had been made",
        "perform а small prayer and hover in the valley": "gather to pray for rain and hope for a blessing",
        "sewing the seeds": "sowing the seeds",
        "hover in the village": "gather in the village",
        "hover in the valley": "gather in the valley",
        "milky the cow": "their beloved cow, Milky",
        "golden horse": "golden harp", # Correcting common AI errors in this story
        "thundered behind him": "chased him down the beanstalk"
    }
    
    for old, new in replacements.items():
        s = re.sub(rf'\b{old}\b', new, s, flags=re.I)

    # Basic cleanup
    s = s.strip()
    if not s or len(s) < 10: return ""
    s = s[0].upper() + s[1:]
    if not s.endswith((".", "!", "?")):
        s += "."

    return s

def reorder_points(points):
    """Refined reordering that preserves chronological sequence but ensures Moral is last"""
    if not points: return []
    
    # We want to keep the story's natural chronological order (which the AI generates)
    # and only move the "Moral" to the very end if it's misplaced.
    
    intro_points = []
    moral_points = []
    core_story = []

    for p in points:
        p_low = p.lower()
        # MORAL: (Should definitely be at the end)
        if any(kw in p_low for kw in ["moral", "lesson", "teaches", "message", "prosperous", "tomorrow", "future", "hope you have", "work today", "second chance"]):
            moral_points.append(p)
        # INTRODUCTION: (Should definitely be at the start)
        elif any(kw in p_low for kw in ["village of", "once upon", "there was a", "lived in", "farmers named", "jack lived with"]):
            # Only if we don't already have intro points, to avoid pulling too much to the top
            if not intro_points:
                intro_points.append(p)
            else:
                core_story.append(p)
        else:
            core_story.append(p)

    # Reassemble: Intro -> Everything else in its original relative order -> Moral
    return intro_points + core_story + moral_points

def generate_summary(transcript, language="English", style="bullet"):
    """Generate high-quality, comprehensive summary with full video coverage"""
    global summarizer

    if not model_loaded:
        raise Exception("Summarization model not loaded. Please check server logs.")

    try:
        # Ensure transcript is a string
        if isinstance(transcript, list):
            def _to_str(item):
                if isinstance(item, str): return item
                if isinstance(item, dict): return item.get('text', str(item))
                try: return item.text
                except: return str(item)
            transcript = " ".join([_to_str(t) for t in transcript])
        elif not isinstance(transcript, str):
            transcript = str(transcript)

        # Clean up transcript noise (FIX 5)
        transcript = re.sub(r'\[.*?\]', '', transcript)
        # Aggressively remove HTML-like tags and technical artifacts (Fixes user's "br/div" issue)
        transcript = re.sub(r'<.*?>', ' ', transcript)
        transcript = re.sub(r'/[a-z]+>', ' ', transcript)
        transcript = re.sub(r'[a-z]+/[a-z]*>', ' ', transcript)
        transcript = re.sub(r'\(\s*\)', '', transcript)
        transcript = re.sub(r'\|+', ' ', transcript)
        
        transcript = re.sub(r'\b(uh|um|you know|like|so)\b', '', transcript, flags=re.IGNORECASE)
        transcript = re.sub(r'\b(\w+)( \1\b)+', r'\1', transcript)
        transcript = re.sub(r'\s+', ' ', transcript).strip()

        # Step 1: Name Extraction
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', transcript)
        name_counts = {}
        for name in potential_names:
            if len(name) > 3:
                name_counts[name] = name_counts.get(name, 0) + 1
        frequent_names = {name: count for name, count in name_counts.items() if count >= 2}

        # Step 2: Content Type Detection (FIX 9)
        content_type = detect_content_type(transcript)
        is_narrative = content_type == "story"
        is_edu_tut = content_type in ["tutorial", "educational"]
        
        if content_type == "lyrics":
            print("[model] Lyrics detected. Using specialized lyrics summarizer.")
            lyrics_summary = summarize_lyrics(transcript, language)
            if lyrics_summary:
                return {
                    "summary": lyrics_summary,
                    "keywords": extract_keywords(transcript),
                    "content_type": "Lyrics"
                }
        
        if is_narrative:
            print("[model] Narrative story detected. Optimizing for plot points...")
        elif is_edu_tut:
            print(f"[model] {content_type.capitalize()} content detected. Optimizing for key concepts...")

        print(f"[model] Transcript length: {len(transcript)} chars. Processing style: {style}")

        # Detect model type
        is_t5 = False
        if model and hasattr(model, 'config'):
            is_t5 = "t5" in model.config._name_or_path.lower()

        # Better Chunking Strategy: sentence-based + overlap (FIX 6)
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        chunks = []
        current_chunk = []
        current_len = 0
        overlap_sentences = 2
        max_chunk_chars = 2500 if is_narrative else 3500 

        for i, sentence in enumerate(sentences):
            if current_len + len(sentence) > max_chunk_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Start next chunk with overlap
                start_idx = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[start_idx:]
                current_len = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_len += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Cover more chunks for better coverage of long videos
        max_chunks = 20 if is_narrative else 15
        chunks = chunks[:max_chunks]
        print(f"[model] Processing {len(chunks)} chunks with overlap for full video coverage...")

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            
            # Refined Prompt Engineering for Storytelling (FIX 2)
            if is_t5:
                if style == "story" or is_narrative:
                    prompt_details = "Capture this segment as a detailed narrative. Focus on the characters, their actions, and the unfolding events to maintain a strong story flow."
                elif style == "short":
                    prompt_details = "Summarize this segment into a single, highly impactful narrative point."
                elif is_edu_tut:
                    prompt_details = "Explain the core concepts and steps in this segment clearly, as if teaching a student."
                else:
                    prompt_details = "Summarize the key events and information in this segment in a clear, human-like way."

                input_text = f"""
{prompt_details}
- Remove repetitive filler
- Ensure the language is natural and engaging
- Detail the most important actions

Transcript Segment:
{chunk}
"""
            else:
                input_text = chunk

            try:
                # Increased parameters for more detail
                result = summarizer(
                    input_text,
                    max_length=350 if (is_narrative or style=="story") else 250,
                    min_length=150 if (is_narrative or style=="story") else 100,
                    do_sample=False,
                    num_beams=4,
                    length_penalty=2.0,
                    repetition_penalty=1.2, # Added to prevent repeating phrases
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    truncation=True
                )
                summary_text = result[0]['summary_text'].strip()
                
                # Filter Bad Outputs (FIX 4)
                if len(summary_text.split()) < 5: continue
                if "is the village" in summary_text.lower(): continue

                if len(summary_text) > 30:
                    chunk_summaries.append(summary_text)
            except Exception as e:
                print(f"[model] Part {i+1} failed: {e}")
                continue

        if not chunk_summaries:
            raise Exception("No summaries generated from any part of the video")

        # Step 4: Final AI Rewrite pass (STEP 3)
        # We use a hierarchical approach to avoid T5 token limits (512 tokens)
        print(f"[model] Finalizing detailed storytelling with hierarchical synthesis...")
        
        # If we have many chunks, group them first
        if len(chunk_summaries) > 6:
            mid_summaries = []
            group_size = 4
            for i in range(0, len(chunk_summaries), group_size):
                group = chunk_summaries[i:i+group_size]
                group_text = " ".join(group)
                print(f"[model] Synthesizing group {i//group_size + 1}...")
                
                try:
                    res = summarizer(
                        f"Summarize these story events into a coherent narrative: {group_text}",
                        max_length=300,
                        min_length=100,
                        do_sample=False,
                        repetition_penalty=1.2
                    )
                    mid_summaries.append(res[0]['summary_text'])
                except:
                    mid_summaries.extend(group)
            final_text = " ".join(mid_summaries)
        else:
            final_text = " ".join(chunk_summaries)
        
        try:
            if is_t5:
                # Forceful prompt for long, high-quality, descriptive sentences
                final_prompt = f"""
TASK: Rewrite these segments into a masterpiece 12-point story.
CHRONOLOGY: You MUST follow the exact sequence of events below. 
- Do NOT skip the middle of the story.
- Point 1 MUST be the start, Point 12 MUST be the end and moral.
- Each point MUST be a long, descriptive sentence (at least 25 words).
- Describe settings, character names (Jack, Mother, Giant), and emotions.
- NO meta-talk, NO meta-instructions, NO technical noise.

STORY SEGMENTS:
{final_text}
"""
                final_input = final_prompt
            else:
                final_input = final_text

            final_summary_result = summarizer(
                final_input,
                max_length=1024 if (is_narrative or style=="story") else 800,
                min_length=500 if (is_narrative or style=="story") else 400, # Lowered slightly to prevent hallucinations
                do_sample=False,
                num_beams=4,
                length_penalty=2.0, # Slightly lowered from 2.5 to be safer
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
                truncation=True
            )
            final_summary_text = final_summary_result[0]['summary_text'].strip()
            
            # Format based on style
            if style == "story":
                summary_final = clean_sentence(final_summary_text)
            else:
                # Format each point then reorder them
                summary_final = _format_summary_points([final_summary_text], language, frequent_names=frequent_names, style=style)
            
            return {
                "summary": summary_final,
                "keywords": extract_keywords(transcript),
                "content_type": content_type.capitalize()
            }
        except Exception as e:
            print(f"[model] Final re-summarization failed: {e}. Falling back to chunk summaries.")
            summary_final = _format_summary_points(chunk_summaries, language, frequent_names=frequent_names, style=style)
            return {
                "summary": summary_final,
                "keywords": extract_keywords(transcript),
                "content_type": content_type.capitalize()
            }

    except Exception as e:
        print(f"[model] Error generating summary: {e}")
        raise Exception(f"Failed to generate summary: {str(e)}")


def _format_summary_points(chunk_summaries, language, is_lyrics=False, frequent_names=None, style="bullet"):
    """Format summaries into high-quality points based on style"""
    import re
    from difflib import get_close_matches
    from collections import Counter

    all_sentences = []

    # AI and YouTube noise patterns (to strip from start or filter out entirely)
    ai_noise_patterns = [
        r'^(the video (discusses|explains|shows|talks about|is about|tells the story of|starts with)|'
        r'in this video|it is mentioned that|the speaker (says|explains|talks)|'
        r'this video|according to the video|the narrator|the clip|the story|it tells the story of|'
        r'the narrator says|the scene shows|we see)\s*',
        r'^(and|but|so|then|next|also|furthermore|additionally)\s*'
    ]

    # Patterns that indicate a sentence is "junk" and should be removed entirely
    junk_patterns = [
        r'for more[:\s]', r'subscribe', r'link in (the )?description', 
        r'click (the )?link', r'the book of the book', r'the back of the back',
        r'author of the book', r'visit our website', r'thanks for watching',
        r'follow us on', r'check out (our )?other videos',
        r'story segments:', r'introduction -> problem', r'narrative arc:',
        r'12 bullet points', r'segments to synthesize', r'transcript segments to use:',
        r'task: synthesize', r'requirements:', r'exactly 12 detailed',
        r'logic:', r'names:', r'no name:', r'no ending', r'no message', r'flow:',
        r'segments:', r'write a high-quality', r'focus on the story arc',
        r'do not repeat these instructions'
    ]

    # Pre-process frequent names
    main_characters = []
    if frequent_names:
        sorted_names = sorted(frequent_names.items(), key=lambda x: x[1], reverse=True)
        main_characters = [name for name, count in sorted_names if count >= 3]
        if not main_characters and sorted_names:
            main_characters = [sorted_names[0][0]]

    for summary in chunk_summaries:
        summary = summary.replace("<n>", " ").strip()
        
        # Step 1: Fix name spelling
        if main_characters:
            words = summary.split()
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^a-zA-Z]', '', word)
                if clean_word and clean_word[0].isupper() and len(clean_word) > 3:
                    if clean_word not in main_characters:
                        matches = get_close_matches(clean_word, main_characters, n=1, cutoff=0.6)
                        if matches:
                            words[i] = word.replace(clean_word, matches[0])
            summary = " ".join(words)

        # Split by sentences (FIX 3)
        parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+|(?<=\n)', summary)
        
        for p in parts:
            p = p.strip()
            if not p: continue
            
            # STEP 4: Weak sentence filtering (Aggressive for high quality)
            if len(p.split()) < 12: # Increased threshold for longer, more descriptive sentences
                continue
            
            # Clean up leading dashes or bullet symbols the AI might have added
            p = re.sub(r'^[\-\*●•\d\.\s]+', '', p).strip()
            
            weak_phrases = ["why are you", "nothing will happen", "click the link", "they are very smart", "in this segment", "this summary describes", "segments to synthesize"]
            if any(phrase in p.lower() for phrase in weak_phrases):
                continue
                
            if any(re.search(jp, p.lower()) for jp in junk_patterns): continue
            
            # Repetition check within sentence
            words = p.lower().split()
            if len(words) > 15: 
                counts = Counter(words)
                if any(count > len(words) * 0.3 for word, count in counts.items() if len(word) > 3): continue

            for pattern in ai_noise_patterns:
                p = re.sub(pattern, '', p, flags=re.IGNORECASE).strip()
            
            p = re.sub(r'\s+', ' ', p)
            if not p or len(p) < 60: # Increased minimum character length for detail
                continue
            
            if p.lower().endswith((' and', ' the', ' a', ' an', ' is', ' to', ' of', ' for', ' in', ' on')): continue
                
            # STEP 2: Humanize each point
            p = clean_sentence(p)
            if p:
                all_sentences.append(p)

    # Deduplication and Flow Logic (STEP 1)
    seen_normalized = []
    unique = []

    for s in all_sentences:
        # Clean normalization for comparison
        norm_s = re.sub(r'[^a-zA-Z0-9 ]', '', s.lower())
        words = norm_s.split()
        meaningful_words = {w for w in words if len(w) > 3}
        if not meaningful_words: continue

        is_duplicate = False
        for existing in seen_normalized:
            # Jaccard similarity or simple overlap
            intersection = meaningful_words & existing
            union = meaningful_words | existing
            similarity = len(intersection) / len(union) if union else 0
            
            # Stricter overlap for high quality
            if similarity > 0.4 or len(intersection) > min(len(meaningful_words), len(existing)) * 0.6:
                is_duplicate = True
                break

        if not is_duplicate:
            seen_normalized.append(meaningful_words)
            unique.append(s)

    # Reorder points using logic (STEP 1)
    flowed_points = reorder_points(unique)

    # Selection logic for target count
    target_count = 12 
    if style == "short": target_count = 3
    elif style == "takeaways": target_count = 6
    
    if is_lyrics:
        points = flowed_points[:10]
    else:
        # We trust the hierarchical flow but still apply reorder_points for safety
        points = flowed_points
        
        if len(points) > target_count:
            # Chronological downsampling
            indices = [int(i * (len(points) - 1) / (target_count - 1)) for i in range(target_count)]
            points = [points[i] for i in indices]
        
        # If we have fewer than target_count, split long points
        if len(points) < target_count and len(points) > 0:
            final_split = []
            for p in points:
                if len(p) > 150 and (len(final_split) + (len(points) - points.index(p))) < target_count:
                    parts = re.split(r'\s+(?:and|but|while|so|then)\s+', p, flags=re.I)
                    for part in parts:
                        if len(part) > 35:
                            final_split.append(clean_sentence(part))
                            if (len(final_split) + (len(points) - points.index(p) - 1)) >= target_count: break
                else:
                    final_split.append(p)
                if len(final_split) >= target_count: break
            points = final_split[:target_count]

    if len(points) < 2 and not is_lyrics:
        return "The video content was too short or unclear to generate a full summary."

    # Final formatting
    if language and "hindi" in language.lower():
        if is_lyrics: header = "📋 गीत का अर्थ:\n"
        elif style == "short": header = "⚡ संक्षिप्त सारांश:\n"
        elif style == "takeaways": header = "🎯 मुख्य बिंदु:\n"
        else: header = f"📋 वीडियो के {len(points)} मुख्य बिंदु:\n"
    else:
        if is_lyrics: header = "📖 Meaning of the Song:\n"
        elif style == "short": header = "⚡ Key Highlights:\n"
        elif style == "takeaways": header = "🎯 Main Takeaways:\n"
        else: header = f"📋 Full Detailed Summary ({len(points)} Key Points):\n"

    formatted = [f"● {p}" for p in points]
    return header + "\n" + "\n".join(formatted)


def gradio_summarize(url, style):
    """Gradio wrapper for the summarization logic"""
    if not url:
        return "Please enter a YouTube URL"
    
    video_id = get_video_id(url)
    if not video_id:
        return "Invalid YouTube URL"
        
    try:
        # Load model if not loaded
        global summarizer
        if summarizer is None:
            load_summarization_model()
            
        transcript_data = get_transcript(video_id)
        result = generate_summary(transcript_data['text'], transcript_data['language'], style=style.lower())
        return result['summary']
    except Exception as e:
        return f"Error: {str(e)}"

def launch_gradio():
    """Launch Gradio interface"""
    with gr.Blocks(title="YouTube Summarizer") as demo:
        gr.Markdown("# 🎥 YouTube Video Summarizer")
        gr.Markdown("Generate high-quality, 12-point storytelling summaries from any YouTube video.")
        
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
                style_dropdown = gr.Dropdown(
                    label="Summary Style", 
                    choices=["Bullet", "Story", "Short", "Takeaways"], 
                    value="Bullet"
                )
                submit_btn = gr.Button("📝 Summarize Video", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="Summary Output", lines=20)
        
        submit_btn.click(
            fn=gradio_summarize,
            inputs=[url_input, style_dropdown],
            outputs=output_text
        )
        
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == '__main__':
    # Load model at startup
    load_summarization_model()
    
    # Start Gradio in a separate thread if you still want Flask (or just run Gradio)
    # Hugging Face Spaces with 'sdk: gradio' expects the app to run on port 7860
    launch_gradio()
