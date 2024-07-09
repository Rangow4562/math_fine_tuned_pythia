# Use the official PyTorch image as a parent image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the container
COPY . .

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app

# Command to run your Python application
CMD ["python", "main.py"]
