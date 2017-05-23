FROM python:2

ADD my_spikefinder.py /

RUN pip install -U numpy==1.11.0 \
				pandas \
				matplotlib \
				Cython \
				requests \
				tensorflow \
				keras

RUN git clone https://github.com/j-friedrich/OASIS.git && cd OASIS && python setup.py build_ext -i
RUN git clone https://github.com/codeneuro/spikefinder-python.git

CMD [ "python", "./my_spikefinder.py", "5" ]
