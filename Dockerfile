FROM python:3.10.8-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy
RUN mkdir ./images

COPY ["predict.py", "kitchenware_app.py", "httprequest.py", "./"]

COPY ["./images/0000.jpg", "./images/0008.jpg", "./images/0015.jpg", "./images/2744.jpg", "./images/3242.jpg", "./images/3247.jpg", "./images/8170.jpg", "./images/"]
COPY ["./images/0022.jpg", "./images/1239.jpg", "./images/3168.jpg", "./images/2103.jpg", "./images/5788.jpg", "./images/7522.jpg", "./images/9374.jpg", "./images/"]
COPY ["./images/0019.jpg", "./images/0967.jpg", "./images/2724.jpg", "./images/3135.jpg", "./images/4673.jpg", "./images/7263.jpg", "./images/9168.jpg", "./images/"]
COPY ["./images/0190.jpg", "./images/0848.jpg", "./images/1739.jpg", "./images/3049.jpg", "./images/4366.jpg", "./images/6106.jpg", "./images/9085.jpg", "./images/"]
COPY ["./images/0136.jpg", "./images/1206.jpg", "./images/2113.jpg", "./images/3833.jpg", "./images/5565.jpg", "./images/7261.jpg", "./images/9271.jpg", "./images/"]
COPY ["./images/0018.jpg", "./images/0510.jpg", "./images/1742.jpg", "./images/2721.jpg", "./images/3277.jpg", "./images/4770.jpg", "./images/8204.jpg", "./images/"]

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "kitchenware_app.py","--server.port=8501", "--server.address=0.0.0.0"]