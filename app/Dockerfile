# Use a slim official Python image (multi-arch friendly)
FROM python:3.10-slim

WORKDIR /app

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8080

# Copy requirements first to leverage build cache
COPY req.txt .

# Upgrade pip, create local wheelhouse to avoid network issues during install
RUN pip install --upgrade pip setuptools wheel && \
    mkdir /wheels && \
    pip download -d /wheels -r req.txt --only-binary=:all: || true && \
    pip install --no-index --find-links=/wheels -r req.txt

# Install procps for pkill command
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

# Copy app sources
COPY . .

# Warm-up run to cache components (like streamlit_option_menu)
RUN streamlit run dashboard.py --server.headless true --server.port 8080 & \
    sleep 20 && pkill streamlit

EXPOSE 8080

# Start Streamlit app
CMD ["streamlit", "run", "dashboard.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
