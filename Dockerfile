FROM python:3.9
WORKDIR /api
ADD . .
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python","api_case.py"]