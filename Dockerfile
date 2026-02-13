FROM --platform=linux/arm64 dailyco/pipecat-base:latest-py3.12

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./bot.py bot.py
