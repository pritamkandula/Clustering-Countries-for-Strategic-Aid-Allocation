FROM python:3.12.0

WORKDIR /projects18/Clustering_Countries

RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENV FLASK_APP=main.py
CMD ["flask", "run", "--host=0.0.0.0"]
