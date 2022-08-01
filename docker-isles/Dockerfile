FROM python:3.9-slim

RUN apt-get update && apt-get -y install --reinstall build-essential && apt-get -y install git && apt-get -y install gcc

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnunet /opt/algorithm/nnunet
COPY --chown=algorithm:algorithm utils /opt/algorithm/utils
COPY --chown=algorithm:algorithm checkpoints /opt/algorithm/checkpoints
COPY --chown=algorithm:algorithm config /opt/algorithm/config

COPY --chown=algorithm:algorithm test /opt/algorithm/test

ENTRYPOINT python -m process $0 $@
