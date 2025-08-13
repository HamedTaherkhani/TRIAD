FROM python:3.10-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements-dl.txt .

# Install dependencies
RUN pip install --no-cache-dir --ignore-installed -r requirements-dl.txt

# Clone the kornia repository
RUN git clone https://github.com/kornia/kornia.git

# Install kornia from source
WORKDIR /app/kornia
RUN pip install -e .

# Verify installation
RUN python -c "import kornia; print(f'Kornia version: {kornia.__version__}')"

# Create a script to handle development workflow
RUN echo '#!/bin/bash\n\
if [ -d "/app/dev-kornia" ]; then\n\
    echo "Development mode: Using mounted code from /app/dev-kornia"\n\
    cd /app/dev-kornia\n\
    pip install -e . --quiet\n\
else\n\
    echo "Using cloned repository"\n\
    cd /app/kornia\n\
fi\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
