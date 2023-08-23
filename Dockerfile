FROM python:3.8-slim

RUN apt update -y && apt install awscli -y

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

RUN pip install --upgrade pip



# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the port that your application will listen on
EXPOSE 8501

# Command to run both the Python script and Streamlit app using do
CMD python App.py & streamlit run --server.port 8501 App.py
