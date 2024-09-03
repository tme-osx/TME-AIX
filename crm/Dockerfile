# Author: Fatih E. NAR
# Telco ChatBot AI App
# Use the official UBI Python image as the base image
FROM registry.access.redhat.com/ubi8/python-39

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# If you don't have a requirements.txt, you can install the required packages directly
# RUN pip install flask speechrecognition gtts requests

# Expose the port the app runs on
EXPOSE 15000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Set environment variables for the API key (optional, better to pass at runtime)
# ENV API_KEY=your_api_key_here

# Command to run the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=15000"]
