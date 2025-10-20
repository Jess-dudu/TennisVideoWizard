FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install uv for manage python packages
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install some needed packages & just
RUN apt-get update && apt-get -y install --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin \
    && rm -rf /var/lib/apt/lists/*
    
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/home/vscode/venv
ENV UV_PROJECT_ENVIRONMENT=/home/vscode/venv
ENV UV_VENV_CLEAR=1
# Place entry points in the environment at the front of the path
ENV PATH="/home/vscode/venv/bin:$PATH"

# USER vscode
# COPY requirements.txt .
# RUN uv venv /home/vscode/venv && uv pip install -r requirements.txt
# RUN uv venv /home/vscode/venv

