# Use Python as a base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /home/dev/fastapi/analytics_app

# Install Poetry
RUN pip install poetry

# Copy Poetry lock files to container
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy the entire application into the working directory
COPY . .

# Expose the application port
EXPOSE 5001

# Command to run the application
CMD ["poetry", "run", "uvicorn", "main:app", "--port=5001", "--workers=2", "--reload", "--host", "0.0.0.0"]