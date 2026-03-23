# 🎥 YouTube Video Summarizer

An AI-powered tool that summarizes YouTube videos into 10 clear, professional points using Google Gemini AI.

## 🚀 Deployment to Render (Recommended)

This project is optimized for a single-service deployment on Render.

### 1. Push to GitHub
- Create a new repository on GitHub.
- Push your code to the repository:
  ```bash
  git init
  git add .
  git commit -m "Ready for deployment"
  git branch -M main
  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
  git push -u origin main
  ```

### 2. Create Render Web Service
- Go to [dashboard.render.com](https://dashboard.render.com).
- Click **New +** → **Web Service**.
- Connect your GitHub repository.
- Use these settings:
  - **Name:** `youtube-summarizer`
  - **Environment:** `Python 3`
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `gunicorn app:app`

### 3. Add Environment Variables
In the **Environment** tab on Render, add:
- `GEMINI_API_KEY`: Your Google AI Studio key.
- `GEMINI_MODEL`: `gemini-2.5-flash` (or your preferred model).

---

## 💻 Local Development

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App:**
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000` in your browser.

## ✨ Features
- **Smart Summarization:** Uses Gemini 2.5 Flash for high-quality summaries.
- **Modern UI:** Balanced, professional design with a clean "job project" vibe.
- **Fast:** Direct transcript extraction and optimized AI prompts.
