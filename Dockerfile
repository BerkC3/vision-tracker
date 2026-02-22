# =============================================================================
# Vision-Track — Multi-stage Dockerfile
# =============================================================================
#
# Two build targets are supported via the TORCH_INDEX_URL build argument:
#
#   CPU (default) — suitable for CI, dev machines, and servers without a GPU:
#     docker build -t vision-track .
#
#   CUDA 12.4 — requires NVIDIA Container Toolkit on the Docker host:
#     docker build \
#       --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#       -t vision-track:cuda .
#
# Override the runtime command to use the CLI instead of the Streamlit UI:
#   docker run vision-track python main.py --source /data/video.mp4 --no-display
# =============================================================================


# =============================================================================
# Stage 1 — builder
# Install all Python dependencies into an isolated virtual environment.
# Using a venv means the runtime stage only needs to COPY /opt/venv; it does
# not need pip, setuptools, or any build-time compilers.
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools needed to compile native extensions (lapx uses C extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Create the virtual environment that will be copied to the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

# ---------------------------------------------------------------------------
# Install PyTorch first, controlled by the TORCH_INDEX_URL build argument.
# This is separated from the rest of requirements.txt so that switching
# between CPU and CUDA builds only affects this single layer.
# ---------------------------------------------------------------------------
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
        "torch>=2.6.0" \
        "torchvision>=0.21.0" \
    --index-url "${TORCH_INDEX_URL}"

# ---------------------------------------------------------------------------
# Install the remaining application dependencies from requirements.txt.
# The grep filter removes:
#   - torch / torchvision  (already installed above)
#   - --extra-index-url    (only needed for local CUDA pip installs)
# This keeps the layer cache valid even when the torch version changes.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|--extra-index-url|psycopg2-binary)' requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Install psycopg2-binary separately — it is optional and skipped in CPU
# builds that use SQLite.  Include it unconditionally so the image works
# with both SQLite and PostgreSQL DATABASE_URL values at runtime.
RUN pip install --no-cache-dir "psycopg2-binary>=2.9.0"


# =============================================================================
# Stage 2 — runtime
# Minimal image: only the venv and the application source are copied in.
# No compilers, no pip, no build artefacts.
# =============================================================================
FROM python:3.11-slim AS runtime

# Runtime libraries required by OpenCV (libgl1) and multi-threaded code.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user — best practice for containerised services.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Bring in the pre-built virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source (respects .dockerignore if present).
COPY --chown=appuser:appuser . .

# Create the output directory for the SQLite database and any saved snapshots.
RUN mkdir -p outputs && chown appuser:appuser outputs

USER appuser

# Streamlit listens on 8501 by default.
EXPOSE 8501

# Verify that Streamlit responds within 30 s of container start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command — serve the Streamlit dashboard in headless mode.
# The --server.headless flag suppresses the "email" prompt on first run.
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
