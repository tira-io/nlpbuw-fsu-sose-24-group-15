FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD module.txt /code/module.txt

RUN pip install -r code/module.txt

ENTRYPOINT [ "python3", "/code/run.py" ]