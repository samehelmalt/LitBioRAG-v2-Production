FROM python:3.11-slim
WORKDIR /srv/aeos
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LITBIORAG_API_HOST=0.0.0.0 LITBIORAG_API_PORT=8471
ARG EXTRAS="index,api,kb,pptx"
ARG WITH_OCR="0"
COPY pyproject.toml README.md ./
COPY app ./app
COPY models ./models
COPY configs ./configs
COPY scripts ./scripts
COPY evaluation/datasets/splits ./evaluation/datasets/splits
RUN if [ "$WITH_OCR" = "1" ]; then apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir pytesseract; fi
RUN pip install --no-cache-dir -e ".[${EXTRAS}]"
# GPU / local-model image: docker build --build-arg EXTRAS="index,api,kb,pptx,ml" -t aeos-medical:ml .
EXPOSE 8471
VOLUME ["/srv/aeos/data"]
CMD ["python", "-m", "scripts.entrypoint"]
