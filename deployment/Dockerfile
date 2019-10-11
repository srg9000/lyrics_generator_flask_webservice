FROM ubuntu:16.04
MAINTAINER Sajan Gohil "sajangohil11@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev --fix-missing
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt
COPY . /app
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
