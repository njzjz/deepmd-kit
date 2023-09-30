FROM python:3.10 AS compile-image
RUN python -m venv /opt/deepmd-kit
# Make sure we use the virtualenv
ENV PATH="/opt/deepmd-kit/bin:$PATH"
# Install package
COPY dist /dist
RUN pip install --pre --find-links /dist deepmd-kit[gpu,cu11,lmp,ipi] \
    && dp -h \
    && lmp -h \
    && dp_ipi \
    && python -m deepmd -h

FROM python:3.10 AS build-image
COPY --from=compile-image /opt/deepmd-kit /opt/deepmd-kit
ENV PATH="/opt/deepmd-kit/bin:$PATH"
CMD ["/bin/bash"]
