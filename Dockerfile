# Use an official lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
# (Choose one requirements file — adjust if needed)
RUN pip install --no-cache-dir -r requirements-mac.txt || \
    pip install --no-cache-dir -r requirements-pi.txt || true

# Optional: set environment variables
ENV PYTHONUNBUFFERED=1

# Expose a port (only if your script runs a server)
EXPOSE 8000

# Run your app
CMD ["python", "run_prod_visual_enhanced.py"]
