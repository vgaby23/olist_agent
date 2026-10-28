FROM python:3.13-slim

WORKDIR /app

# Copy the code and the raw CSV files
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Build dataset/olist.db, then remove the CSV files to keep the image smaller
RUN python dataset/preprocessing/numeric_data/clean_olist.py \
    && rm -rf dataset/raw_data

# Make port 8000 available to the world outside this container
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
