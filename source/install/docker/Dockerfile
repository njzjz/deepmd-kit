FROM python:3.11 AS compile-image
ARG VARIANT=""
ARG CUDA_VERSION="12"
RUN python -m pip install uv
RUN python -m uv venv /opt/deepmd-kit
# Make sure we use the virtualenv
ENV PATH="/opt/deepmd-kit/bin:$PATH"
ENV VIRTUAL_ENV="/opt/deepmd-kit"
# Install package
COPY dist /dist
RUN if [ "${CUDA_VERSION}" = 11 ]; then export UV_TORCH_BACKEND=cu118; fi \
    && uv pip install "$(ls /dist/deepmd_kit${VARIANT}-*manylinux*_x86_64.whl)[gpu,cu${CUDA_VERSION},lmp,ipi,torch]" \
    && dp -h \
    && lmp -h \
    && dp_ipi \
    && python -m deepmd -h

FROM python:3.11 AS build-image
COPY --from=compile-image /opt/deepmd-kit /opt/deepmd-kit
ENV PATH="/opt/deepmd-kit/bin:$PATH"
CMD ["/bin/bash"]
