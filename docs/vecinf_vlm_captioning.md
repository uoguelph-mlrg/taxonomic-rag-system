# Vec-Inf VLM Captioning

This workflow deploys `Qwen2.5-VL-7B-Instruct` with `vec-inf`, generates
rare-species image captions through the OpenAI-compatible VLM endpoint, writes
RAG-ready caption JSONL, and shuts down the deployed model server.

## Smoke Test

Run this from the cluster login node after confirming `vec-inf` is available in
the project environment:

```bash
export VEC_INF_ACCOUNT=aip-your-account
export VEC_INF_WORK_DIR=/path/to/vec-inf-workdir

INTERVAL_START=0 \
INTERVAL_END=3 \
BATCH_SIZE=1 \
MAX_TOKENS=1024 \
bash submit_vecinf_vlm_caption_pipeline.sh
```

The wrapper jobs are CPU-only. The GPU allocation is created by `vec-inf launch`
for the model server.

## Expected Artifacts

The pipeline writes outputs under:

```text
outputs/vecinf_vlm_caption/<RUN_ID>/
```

Important files:

- `state/vlm_launch_response.json`: raw `vec-inf launch --json-mode` response.
- `state/vlm_server_job_id.txt`: Slurm job ID for the deployed model server.
- `RS_vecinf_qwen25vl_captions_<timestamp>.jsonl`: one caption record per sample.
- `RS_vecinf_qwen25vl_captions_<timestamp>_metadata.json`: run metadata.

Each JSONL record includes `rsid`, `caption`, `true_class`, `model`, `base_url`,
`status`, `error`, interval bounds, and `created_at`. A successful smoke test
should have non-empty captions and `status` set to `ok`.

## Cleanup

The pipeline submits a shutdown stage with `afterany`, so the vec-inf model
server should be shut down whether captioning succeeds or fails. If needed,
manually shut down the server with:

```bash
vec-inf shutdown "$(tr -d '[:space:]' < outputs/vecinf_vlm_caption/<RUN_ID>/state/vlm_server_job_id.txt)"
```
