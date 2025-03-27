FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required by GeoPandas
RUN apt-get update && apt-get install -y \
    build-essential \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create data directory
RUN mkdir -p /app/data

# Expose the port
EXPOSE 7868

# Command to run app
CMD ["python", "app.py"]
