# Pixel_Triage

Transparent digital forensics engine that harvests, compiles, and visualizes multi-layered technical evidence to detect synthetic media and empower human judgment.

## Prerequisites
- **Docker** (recommended) **or** **Python 3.10+** and `pip` (for local run)  
- **Git** to clone the repository

## Using Docker

```bash
# Clone the repo
git clone https://github.com/manojkumar377/Pixel_Triage_
cd Pixel_Triage_

# Build the Docker image
docker build -t ai-forensics .

# Run the container (maps port 8000)
docker run --rm -p 8000:8000 ai-forensics

http://127.0.0.1:8000/ - Open this in your browser
