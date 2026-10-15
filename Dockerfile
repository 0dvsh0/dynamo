# Dynamo — headless AI Visual & Video Harness
#
#   docker build -t dynamo .
#   docker run --rm -v "$PWD/output:/out" dynamo -- python agent.py
#
# Dynamo manages Xvfb itself: no DISPLAY export, no waiting for X to come up.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb ffmpeg x11-utils x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir .

# Fail the build rather than the run if a dependency is missing.
RUN dynamo info || (echo "dynamo cannot see ffmpeg/Xvfb" && exit 1)

WORKDIR /app
ENTRYPOINT ["dynamo", "record", "--fps", "24", "--session", "/out/session", "--"]
CMD ["python", "-c", "print('nothing to run: pass your agent after --')"]
