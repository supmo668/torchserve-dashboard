
FROM pytorch/torchserve:latest-gpu

RUN python -m pip install -U pip && \
    pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple -U ezout-yolovision
    
COPY modelstore/* /home/model-server/model-store/
COPY config/* /home/model-server/config/

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]

EXPOSE 8080
EXPOSE 8081

CMD ["serve"]