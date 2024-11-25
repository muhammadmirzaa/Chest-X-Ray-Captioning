# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Install the environment
RUN conda env create -f environment.yml || tail -f /dev/null

# Activate the environment and ensure it's added to the PATH
RUN echo "source activate chexpert" > ~/.bashrc
ENV PATH /opt/conda/envs/chexpert/bin:$PATH

# Make port 8501 available to the world outside this container (default port for Streamlit)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
