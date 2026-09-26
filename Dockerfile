FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    DEVICE=cpu \
    HF_HOME=/root/.cache/huggingface

# Set working directory
WORKDIR /workspace

# Install Python dependencies from wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Default entrypoint / keep-alive command
CMD ["tail", "-f", "/dev/null"]
