FROM dailyco/pipecat-base:0.1.14
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
ARG CACHEBUST=1
COPY ./bot.py bot.py
CMD ["python", "bot.py"]