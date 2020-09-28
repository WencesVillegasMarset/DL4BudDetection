FROM python:3.7.9

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

RUN echo "deb http://httpredir.debian.org/debian jessie main contrib" > /etc/apt/sources.list \
    && echo "deb http://security.debian.org/ jessie/updates main contrib" >> /etc/apt/sources.list \
    && echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections \
    && apt-get update && apt-get install -y ttf-mscorefonts-installer

# Set work directory

WORKDIR /figures

# Install dependencies

COPY requirements.txt /figures/

RUN pip install --no-cache-dir -r requirements.txt

# Copy files

COPY . /figures/

CMD ["python", "generate_plots.py"]

