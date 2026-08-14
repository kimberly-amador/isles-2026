# Adapted from the ISLES'26 algorithm template:
# https://github.com/ezequieldlrosa/isles26-docker-template
#
# Changes from the template: nnU-Net in requirements, our postprocessing module
# copied in, and the nnUNet_* env vars pointed at /tmp (the only writable path
# outside the mounts).
FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles26_algorithm_amd64
# cuda12.6 is what Grand Challenge's T4 instances need, and it happens to match
# the torch our checkpoints were trained under on ARC (2.13.0+cu126) -- so the
# weights load without a version skew.

ENV PYTHONUNBUFFERED=1
# nnU-Net reads these at import and writes scratch under them. Grand Challenge
# allows writes only to the mounted dirs and /tmp.
ENV nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Inherit torch and the CUDA libs from the base image rather than reinstalling
# them; --without-pip because the base ships no ensurepip.
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
# The exact module the CV scores were produced with -- imported, not reimplemented,
# so the shipped floor cannot drift from the evaluated one.
COPY --chown=user:user postprocessing.py /opt/app/

# Required: Grand Challenge uses this label to detect the invoke API.
LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
