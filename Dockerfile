FROM python

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY multiagents2.py multiagents2.py

COPY .env .env

CMD ["python3", "multiagents2.py"]