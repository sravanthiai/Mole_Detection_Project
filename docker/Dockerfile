FROM python



RUN mkdir /app
COPY requirements.txt .
COPY app.py /app/app.py

COPY . /app

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN pip3 install pillow

WORKDIR /app

CMD ["python3", "app.py"]