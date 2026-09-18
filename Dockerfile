# The geospatial wheels (geopandas, pyogrio, shapely, pyproj, rasterio) bundle GDAL/GEOS/PROJ,
# so a slim Debian base needs no system GDAL and no compiler. Do not switch this to Alpine:
# musl has no manylinux wheels and every one of those would fall back to a source build.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DARKVESSEL_CONFIG=configs/pipeline.yaml

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[web]"

COPY configs ./configs

# Bake the no-network fixture so the container has something to show on first start.
RUN darkvessel synthesise --out data/synthetic

RUN useradd --create-home --uid 10001 darkvessel && chown -R darkvessel /app
USER darkvessel

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

# Platforms that inject $PORT (Render, Railway, Cloud Run) override this with their own command.
CMD ["uvicorn", "darkvessel.web.app:app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
