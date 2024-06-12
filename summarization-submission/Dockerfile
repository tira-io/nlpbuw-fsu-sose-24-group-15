FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD requirements.txt /code/requirements.txt

RUN pip install -r code/requirements.txt

ENTRYPOINT [ "python3", "/code/run.py" ]