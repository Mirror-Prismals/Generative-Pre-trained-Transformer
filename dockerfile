# ─── Base image: CUDA 12.1 runtime + PyTorch 2.3 ──────────────────────────────
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Avoid tzdata prompts in Debian images
ARG DEBIAN_FRONTEND=noninteractive

# ─── Python deps not in the base image ───────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/reqs.txt

# copy requirements after installing – keeps layers tidy
COPY requirements.txt /tmp/reqs.txt

# ─── Application code ───────────────────────────────────────────────────────
WORKDIR /workspace
COPY gpt_text_trainer.py .

# Nice defaults; can be overridden at run-time via CLI flags
ENV TEXT_CORPUS=/data/text \
    VOCAB_SRC=/data/text \
    CKPT=/workspace/gpt_text.pt

# Show help if no args; otherwise pass straight through
ENTRYPOINT ["python", "gpt_text_trainer.py"]
CMD ["--help"]
