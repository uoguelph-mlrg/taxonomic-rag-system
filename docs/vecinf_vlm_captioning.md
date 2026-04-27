# Vec-Inf VLM Captioning

This workflow deploys `Qwen2.5-VL-7B-Instruct` with `vec-inf`, generates
rare-species image captions through the OpenAI-compatible VLM endpoint, writes
RAG-ready caption JSONL, and shuts down the deployed model server.

## Smoke Test

Run this from the cluster login node after confirming `vec-inf` is available in
the project environment. Use a checkout under **`/scratch`** or **`/project`**
(not `/home/...`, including symlinked “scratch” under home), because Slurm may
reject nested `sbatch` calls when the working tree is considered under `/home`.

Defaults assume the repo lives at `/scratch/${USER}/taxonomic-rag-system` and
that `VEC_INF_WORK_DIR` matches unless you override it.

```bash
cd /scratch/${USER}/taxonomic-rag-system
export VEC_INF_ACCOUNT=aip-your-account
# Optional if your clone is elsewhere:
# export REPO_DIR=/scratch/${USER}/taxonomic-rag-system
# export VEC_INF_WORK_DIR="$REPO_DIR"

INTERVAL_START=0 \
INTERVAL_END=3 \
BATCH_SIZE=1 \
MAX_TOKENS=1024 \
bash slurm_bash_vector-inf/submit_vecinf_vlm_caption_pipeline.sh
```

The wrapper jobs are CPU-only. The GPU allocation is created by `vec-inf launch`
for the model server.

**Stage 1** runs `vec-inf launch`, then **blocks** until the deployed server reports
READY (via `VecInfClient.wait_until_ready`) and writes `state/vlm_base_url.txt`.
**Stage 2** is submitted with `afterok:stage1`, so captioning starts only after
that wait succeeds. Stage 2 **requires** `state/vlm_base_url.txt` and always
passes `--base-url` to the Python runner; the caption script does not call vec-inf
or `wait_until_ready` (only Stage 1 does).

**`VEC_INF_READY_TIMEOUT_SECONDS`** (default `14400`, four hours) caps how long
Stage 1 will poll for readiness. If you raise it above the Stage 1 Slurm
walltime (`#SBATCH --time` in `submit_vecinf_vlm_stage1_launch.sh`, currently
five hours), increase the walltime so the job is not killed before the wait
finishes.

## Expected Artifacts

The pipeline writes outputs under:

```text
outputs/vecinf_vlm_caption/<RUN_ID>/
```

Important files:

- `state/vlm_launch_response.json`: raw `vec-inf launch --json-mode` response.
- `state/vlm_server_job_id.txt`: Slurm job ID for the deployed model server.
- `state/vlm_base_url.txt`: OpenAI-compatible `base_url` written after the server
  reaches READY (**required** by Stage 2; the job fails if this file is missing
  or empty).
- `RS_vecinf_qwen25vl_captions_<timestamp>.jsonl`: one caption record per sample.
- `RS_vecinf_qwen25vl_captions_<timestamp>_metadata.json`: run metadata.

Each JSONL record includes `rsid`, `caption`, `true_class`, `model`, `base_url`,
`status`, `error`, interval bounds, and `created_at`. A successful smoke test
should have non-empty captions and `status` set to `ok`.

## Re-running caption only

To run [`rare_species_caption_vecinf_vlm.py`](src/taxonomic_rag_system/runs/rare_species_caption_vecinf_vlm.py)
manually, you must pass a non-empty `--base-url` (for example the first line of
`state/vlm_base_url.txt`). `--server-job-id` is optional and used for metadata
only. Resolving a URL from a job id inside Python is not supported.

## Cleanup

The pipeline submits a shutdown stage with `afterany`, so the vec-inf model
server should be shut down whether captioning succeeds or fails. If needed,
manually shut down the server with:

```bash
vec-inf shutdown "$(tr -d '[:space:]' < outputs/vecinf_vlm_caption/<RUN_ID>/state/vlm_server_job_id.txt)"
```
