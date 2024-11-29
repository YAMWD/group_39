FROM debian:stable-slim
RUN apt-get update && apt-get -y upgrade && apt-get install -y python3 python3-dev python3-venv vim sudo cmake g++

RUN useradd -ms /bin/bash --create-home user && usermod -aG sudo user
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user
WORKDIR /home/user
COPY --chown=user:user files .

RUN python3 -m venv venv
ENV VIRTUAL_ENV /home/user/venv
ENV PATH /home/user/venv/bin:$PATH

RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
