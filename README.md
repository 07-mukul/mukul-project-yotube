---
title: Youtube Summarizer
emoji: 🎥
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# YouTube Summarizer

A high-quality YouTube video summarizer using FLAN-T5-Base.

## Features
- **12-Point Detailed Summary**: Generates a comprehensive narrative of the video.
- **Story Arc Preservation**: Follows the natural chronological flow of the video.
- **Human-Like Language**: Polished output using advanced prompt engineering.
- **Multiple Styles**: Supports Bullets, Story, Short, and Takeaways.

## Deployment on Hugging Face Spaces
This project is configured for deployment using Docker. 

1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Upload all files from this repository (including `Dockerfile`, `app.py`, `requirements.txt`, etc.).
4. Hugging Face will automatically build the container and start the server on port 7860.

## API Endpoints
- `GET /summary?url=VIDEO_URL&style=bullet`: Get a summary of a YouTube video.
