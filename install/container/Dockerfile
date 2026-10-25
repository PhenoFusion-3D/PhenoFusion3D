FROM python:3.11.14-slim-bookworm
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="PhenoFusion3D recorded-data analysis" \
      org.opencontainers.image.licenses="MPL-2.0" \
      org.opencontainers.image.revision=$SOURCE_REVISION
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen QTWEBENGINE_DISABLE_SANDBOX=0 \
    MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libglib2.0-0 libgomp1 libnss3 libdbus-1-3 \
    libfontconfig1 libfreetype6 libxrender1 libsm6 libice6 \
    libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libasound2 libxcomposite1 \
    libxdamage1 libxrandr2 libxtst6 libxss1 libxi6 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/phenofusion3d
COPY install/container/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY pyproject.toml README.md LICENSE main.py ./
COPY app ./app
COPY capture ./capture
COPY file_io ./file_io
COPY processing ./processing
COPY visualiser ./visualiser
COPY tools/handover ./tools/handover
RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip freeze > environment.txt \
    && useradd --create-home --uid 1000 researcher \
    && mkdir -p /data /results && chown researcher:researcher /results
USER researcher
CMD ["python", "tools/handover/container_smoke.py"]
