# Use Python 3.9 as base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip3 install -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 user
USER user

# Environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy files with proper ownership
COPY --chown=user . $HOME/app

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
