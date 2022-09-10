FROM python:3.10.7-bullseye

ADD ./. /growingspheres/
WORKDIR /growingspheres/benchmark/
RUN python3 -m pip install -r requirements_exp.txt

CMD python3 run_exp.py && until python3 -c "from cfbench.cfbench import analyze_results; analyze_results('growingspheres')"; do sleep 10; done