# Multi-stage: build the Vite frontend, then serve static assets + API on one origin.
FROM node:22-slim AS web
WORKDIR /build
COPY inference-sizer/web/package.json inference-sizer/web/package-lock.json ./
RUN npm ci
COPY inference-sizer/web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY inference-sizer/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY inference-sizer/ ./
COPY --from=web /build/dist ./web/dist
# CORS_ORIGINS: comma-separated allowed origins; empty = same-origin only.
ENV CORS_ORIGINS=""
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]