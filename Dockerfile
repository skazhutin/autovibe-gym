FROM booml-backend:latest

WORKDIR /autovibe

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
CMD ["-m", "experiments.run_gym", "--help"]
