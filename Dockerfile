FROM python:3.11-slim
WORKDIR /srv/aeos
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LITBIORAG_API_HOST=0.0.0.0 LITBIORAG_API_PORT=8000
COPY pyproject.toml README.md ./
COPY app ./app
COPY models ./models
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir -e ".[index,api,kb,pptx]"
# ML models (torch, transformers, sentence-transformers) are installed in the GPU image:
#   docker build --build-arg EXTRAS="index,api,kb,pptx,ml" ...
EXPOSE 8000
CMD ["python", "-m", "scripts.serve", "--host", "0.0.0.0", "--port", "8000"]
