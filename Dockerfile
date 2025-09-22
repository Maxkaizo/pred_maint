FROM continuumio/miniconda3

WORKDIR /app

# Copy environment file and install
COPY environment.yml .
RUN conda env create -f environment.yml -y && \
    conda clean --all --yes

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["./entrypoint.sh"]