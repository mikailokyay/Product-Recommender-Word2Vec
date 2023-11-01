FROM python:3.9
WORKDIR /code

# Copy the API code to the container
COPY requirements.txt  /code


RUN pip3 install --no-cache-dir -r /code/requirements.txt


COPY ./ /code

WORKDIR /code

# Define the command to start the API
CMD ["uvicorn", "api_case:app", "--host", "0.0.0.0", "--port", "5000"]