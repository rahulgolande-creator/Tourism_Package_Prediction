# Use Python 3.9 as base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy all project files with correct ownership
COPY --chown=user . $HOME/app

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
