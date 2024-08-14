# Use the UBI8 base image
FROM registry.access.redhat.com/ubi8/python-39

# Set environment variables
ENV APP_HOME=/opt/app-root/src
WORKDIR $APP_HOME

# Copy the Flask app code into the container
COPY chatgpt_flask_app.py .

# Install necessary Python packages
RUN pip install --upgrade pip && \
    pip install flask requests

# Expose the port the app runs on
EXPOSE 5000

# Set the entrypoint for the container
ENTRYPOINT ["python3"]

# Run the Flask app
CMD ["chatgpt_flask_app.py"]
