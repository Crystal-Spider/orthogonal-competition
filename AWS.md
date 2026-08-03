Running the evaluation on AWS EC2
================================

With `backend = "aws"` the evaluator provisions **one** EC2 instance, runs every
(team × dataset × scenario) container on it, and terminates it when done.
Datasets are downloaded onto the instance over HTTP from `datasets_base_url`;
locally-built images are shipped across with `docker save` / `docker load`.

This document covers instance selection, in particular for testing the GPU
execution pipeline. For the runner's implementation see `runner.py`.


Choosing an instance type
-------------------------

### CPU-only runs

Any general-purpose or compute-optimised type works; `c7i.4xlarge` is a
reasonable default. Size the instance by the *datasets*, not the algorithms:
the private datasets plus the `memory` scenario are what drive RAM.

### GPU runs

**`g4dn.xlarge` is the cheapest sensible option for testing the GPU pipeline.**
1× NVIDIA T4 (16 GB VRAM), 4 vCPU, 16 GB RAM.

| Instance       | GPU          | VRAM  | vCPU | RAM   | ~on-demand | When to use |
|----------------|--------------|-------|------|-------|------------|-------------|
| `g4dn.xlarge`  | T4 (sm_75)   | 16 GB | 4    | 16 GB | ~$0.53/hr  | Pipeline smoke tests — cheapest CUDA option |
| `g6.xlarge`    | L4 (sm_89)   | 24 GB | 4    | 16 GB | ~$0.80/hr  | Image needs a newer arch than Turing |
| `g5.xlarge`    | A10G (sm_86) | 24 GB | 4    | 16 GB | ~$1.00/hr  | L4 not available in the region |
| `g6e.xlarge`   | L40S         | 48 GB | 4    | 32 GB | ~$1.86/hr  | Large-VRAM workloads only |

Prices are us-east-1 on-demand and are **indicative only** — check the current
price for your region before launching (eu-west-1 runs a few percent higher).
Spot instances typically cut these by 60–70%, which is worth it for test runs
where an interruption just means re-running.

Avoid:

- **`g4ad.*`** — AMD Radeon Pro V520, no CUDA.
- **`g5g.*`** — Graviton/arm64; the competitors' images are x86_64.
- **`p3.*`** — V100, both older and several times more expensive than the above.

Two sizing notes:

- The `.xlarge` sizes have only **16 GB of host RAM**. That is fine for a
  `fashion-mnist` smoke test, but the private datasets — especially under the
  `memory` scenario — will likely need `g4dn.2xlarge` (32 GB) or larger.
- The T4 is Turing (sm_75). Current PyTorch still supports it, but an image
  built exclusively for newer architectures, or one using FP8, will not run.
  Move to `g6.xlarge` (L4, sm_89) in that case.


Choosing an AMI
---------------

`_DEFAULT_USER_DATA` in `runner.py` installs only Docker and socat — **it does
not install NVIDIA drivers or the container toolkit**. On a plain Ubuntu AMI the
container's `device_requests` will therefore fail.

For GPU runs use the **AWS Deep Learning Base GPU AMI (Ubuntu 22.04)**, which
ships the driver, Docker and `nvidia-container-toolkit` preinstalled, and whose
default user is `ubuntu` (what the runner expects). AMI IDs are region-specific.
The published SSM parameter names change between AMI generations, so list them
first and then resolve the one you want:

```
# find the current parameter name
aws ssm get-parameters-by-path --region eu-west-1 \
  --path /aws/service/deeplearning/ami --recursive \
  --query "Parameters[?contains(Name, 'gpu')].Name" --output text

# resolve it to an AMI id
aws ssm get-parameter --region eu-west-1 --name <the-parameter-name> \
  --query Parameter.Value --output text
```

(The console's AMI catalogue, filtered to "Deep Learning Base GPU", works too.)

The alternative is to keep a plain Ubuntu AMI and supply a custom `user_data`
in the `[runner]` block that installs the driver and toolkit itself; that costs
several minutes of boot time on every run.

Because the DL AMI is large and the datasets are too, raise `volume_size_gb`
well above its default of 100.


Configuration
-------------

A GPU-capable `[runner]` block:

```toml
gpu = true                          # expose host GPUs to the containers

[runner]
backend        = "aws"
instance_type  = "g4dn.xlarge"
region         = "eu-west-1"
ami            = "ami-xxxxxxxx"     # Deep Learning Base GPU AMI, region-specific
key_name       = "orthogonal"       # existing EC2 key pair
key_path       = "~/.ssh/orthogonal.pem"
security_group_ids = ["sg-xxxx"]    # must allow inbound SSH from where the evaluator runs
volume_size_gb = 250
terminate_after = true              # set false to keep the instance for debugging
# datasets_base_url = "https://www.dei.unipd.it/~ceccarello/orthogonal-datasets"
```

The backend can be overridden per invocation with `--backend aws` /
`--backend local`, which is the easy way to keep one config and switch.

Note that `gpu` is read at the **top level** of the config, not per team — a
`gpu = true` inside a `[[teams]]` table is ignored.


Resource measurement
--------------------

Peak RAM and peak VRAM are both measured **on the instance**, via
`runner.host_exec` (see `ContainerResourceMonitor` in `evaluator.py`):

- RAM comes from the container's cgroup (`memory.peak` /
  `memory.max_usage_in_bytes`, falling back to `memory.current`).
- VRAM is the summed `nvidia-smi --query-compute-apps` usage of the processes
  in that same cgroup, so other GPU activity on the host is excluded.

Remote polling runs every 2s rather than 0.5s, since each tick is an SSH round
trip. This does not affect RAM accuracy where the kernel maintains the peak.

If `peak_vram_mb` comes back as 0 for a run you expect to use the GPU, check in
this order: the AMI actually has `nvidia-smi`; the container was started with
`gpu = true`; and the image's CUDA build supports the card's architecture.


Cost control
------------

- `terminate_after = true` (the default) terminates the instance at the end of
  the run, **including** when setup fails partway through.
- With `terminate_after = false` the instance is left running deliberately, and
  the evaluator logs a warning saying so. Remember to terminate it yourself.
- A crash of the evaluator process itself — or losing the SSH connection — can
  still leak a running instance. After GPU test runs, confirm with:

  ```
  aws ec2 describe-instances --region eu-west-1 \
    --filters "Name=tag:Name,Values=orthogonal-competition-runner" \
              "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]" --output table
  ```

  The runner tags every instance it launches `Name=orthogonal-competition-runner`.
