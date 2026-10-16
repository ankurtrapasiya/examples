# Everything the blog's scripts need, installed once, so each run is one command.
#   docker build -t blog-examples .
#   docker run --rm -v "$PWD/charts:/examples/charts" blog-examples scoring-a-model
FROM python:3.12-slim

WORKDIR /examples
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "run.py"]
