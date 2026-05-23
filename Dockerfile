FROM python:3.12-slim
ARG TARGETARCH
ARG TARGETVARIANT

ENV OJ_ENV production
WORKDIR /app

COPY ./deploy/requirements.txt /app/deploy/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    <<EOS
set -ex
sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources
sed -i 's|http://security.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
apt-get update
apt-get install -y --no-install-recommends \
    build-essential libpq-dev libjpeg-dev zlib1g-dev libfreetype-dev \
    supervisor nginx curl unzip ca-certificates
pip install -r /app/deploy/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
apt-get remove -y build-essential libpq-dev libjpeg-dev zlib1g-dev libfreetype-dev
rm -rf /var/lib/apt/lists/*
EOS

COPY ./ /app/
COPY frontend_dist /app/dist

RUN chmod -R u=rwX,go=rX ./ && chmod +x ./deploy/entrypoint.sh

HEALTHCHECK --interval=5s CMD [ "/usr/local/bin/python3", "/app/deploy/health_check.py" ]
EXPOSE 8000
ENTRYPOINT [ "/app/deploy/entrypoint.sh" ]