# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 3000

# Define environment variable to specify how Flask should run
#ENV FLASK_APP=app.py
#ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py when the container launches
CMD ["python", "main.py"]