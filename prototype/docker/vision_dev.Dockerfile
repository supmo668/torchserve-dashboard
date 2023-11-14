#FROM python:3.10-slim-bullseye AS builder
FROM supermomo668/dev_projects:ezout_vision AS builder

# Set the working directory
COPY ./EzOutVision /EzOutVision
WORKDIR /EzOutVision
RUN apt update && apt install python3-opencv libzbar0 -y 

FROM builder as runner
RUN python -m pip install -U pip && pip install -r requirements.txt

EXPOSE 5000 80

#CMD [/bin/bash]
ENTRYPOINT ["gunicorn", "wsgi:app", "-c", "gunicorn.conf.py"]

