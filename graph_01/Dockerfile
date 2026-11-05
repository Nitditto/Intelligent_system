# One image for the loader (kg/load.py) and the API (api/): only the packages they need.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir neo4j fastapi "uvicorn[standard]"
COPY api ./api
COPY kg ./kg
COPY dataset/processed ./dataset/processed
ENV PYTHONUTF8=1 NEO4J_URI=bolt://neo4j:7687
