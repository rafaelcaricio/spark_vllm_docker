# Agent Notes

## Working unholy-fusion DeepSeek V4 Flash setup

Last confirmed working: 2026-06-10.

This repo has a working 2x DGX Spark / GB10 `unholy-fusion` path for
DeepSeek-V4-Flash using:

- Image: `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`
- Served model name: `deepseek-v4-flash`
- Head container: `vllm-spark-head`
- Worker container: `vllm-spark-worker`
- Worker host: `192.168.250.13`
- API endpoint: `http://127.0.0.1:8000/v1/chat/completions`
- Compose override: `compose/docker-compose.unholy.yml`
- Entrypoint: `entrypoints/entrypoint.unholy.sh`
- Expected env file: `.env.unholy-fusion`

Important operational fixes:

- Mount the full Hugging Face model cache repository, not only the snapshot
  directory. The snapshot contains symlinks, so mounting only the snapshot can
  leave model files missing inside the container.
- `entrypoints/entrypoint.unholy.sh` intentionally unsets empty `VLLM_*`
  environment variables before starting vLLM. The unholy-fusion vLLM build
  parses some optional env vars strictly; empty strings caused startup config
  failures.
- The unholy-fusion path is `mp` only. Keep `DISTRIBUTED_BACKEND=mp`.
- First cold startup can spend several minutes compiling DeepGEMM/NVCC kernels
  and capturing CUDA graphs. A log line about no shared memory broadcast block
  for 60 seconds can be benign during compilation.
- The server is ready only after `/health` returns 200 and logs show
  `Application startup complete`.

Validation commands:

```bash
curl -sS --max-time 2 http://127.0.0.1:8000/health
```

```bash
curl -sS --max-time 120 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say hello and confirm you are running."}],"max_tokens":64,"temperature":0.2}'
```

Useful status checks:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
ssh -o BatchMode=yes 192.168.250.13 'docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" && df -h /'
docker logs --since 60s vllm-spark-head
ssh -o BatchMode=yes 192.168.250.13 'docker logs --since 60s vllm-spark-worker'
```

Disk state after the successful run:

- Local `/`: about 101G free.
- Worker `/`: about 309G free after safe Docker cleanup.
- Safe cleanup used stopped-container / dangling-image style pruning only; do
  not run broad destructive cleanup such as `docker image prune -a` unless the
  user explicitly approves it.
