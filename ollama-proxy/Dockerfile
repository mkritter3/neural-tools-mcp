FROM python:3.11-slim

WORKDIR /app

# Install required packages
RUN pip install fastapi uvicorn httpx

# Copy proxy script
COPY proxy.py /app/proxy.py

# Expose port
EXPOSE 11434

# Run the proxy
CMD ["python", "proxy.py"]