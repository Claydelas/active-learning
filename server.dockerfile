FROM python:3.8-slim
COPY backend /app
COPY data /data
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["main.py"]