FROM ubuntu:22.04

WORKDIR /amr_prj

# copy from everything from the working directory + requirements.txt
# COPY requirements.txt /amr_prj/requirements.txt
COPY . /amr_prj

# run some commands
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r /amr_prj/requirements.txt

RUN mkdir -p /amr_prj

CMD ["bash"]