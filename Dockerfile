# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Pre-download the model to avoid downloading at runtime
RUN python download_model.py

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# Command to run the application using gunicorn
# Hugging Face Spaces uses port 7860 by default
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120"]
