FROM node:22-bookworm-slim AS nodejs

FROM nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

COPY --from=nodejs /usr/local /usr/local

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bubblewrap \
        ca-certificates \
        curl \
        git \
        openssh-client \
        python3 \
        python3-pip \
        python3-venv \
        sudo \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g context-mode

RUN if ! getent group "${USER_GID}" >/dev/null; then groupadd --gid "${USER_GID}" "${USERNAME}"; fi \
    && if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
        EXISTING=$(id -nu "${USER_UID}" 2>/dev/null); \
        if [ -n "$EXISTING" ] && [ "$EXISTING" != "${USERNAME}" ]; then \
            usermod -l "${USERNAME}" "$EXISTING" \
            && usermod -d "/home/${USERNAME}" -m "${USERNAME}" \
            && groupmod -n "${USERNAME}" "$EXISTING" 2>/dev/null || true; \
        else \
            useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" --shell /bin/bash; \
        fi; \
    fi \
    && printf '%s ALL=(ALL) NOPASSWD:ALL\n' "${USERNAME}" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN uv pip install --system --no-cache \
    "huggingface_hub[hf_transfer]"

RUN mkdir -p "/home/${USERNAME}/.cache/huggingface" \
    && chown -R "${USERNAME}:${USERNAME}" "/home/${USERNAME}/.cache"

USER ${USERNAME}
WORKDIR /workspaces
