Orthogonal school competition
=============================

This is the evaluation harness for the second year's competition of the
[Orthogonal school](https://www.elicsir.it/en/orthogonal-school), focusing
on k-nearest neighbors search.


Submission format
-----------------

A submission consists of a docker image that extends the
[base image](https://github.com/Cecca/orthogonal-competition/blob/main/Dockerfile)
of the competition. Such image should should contain two files:

- `/app/algorithm.py`: this is a python module defining an `Algorithm` class with
  three methods:
    - `fit` to train the data structure on the dataset to be queried
    - `query` to answer a single k-nn query
    - `get_n_distances` to return the number of distances computed since the instantiation
      of the algorithm
- `/app/scenarios.yaml`: a configuration file that specifies the parameters of the algorithm
  for each execution scenario and, possibly, dataset.

The `template` directory gives a customizable template for the submissions. In particular,
[`algorithm.py`](https://github.com/Cecca/orthogonal-competition/blob/main/template/algorithm.py)
and
[`scenarios.yaml`](https://github.com/Cecca/orthogonal-competition/blob/main/template/scenarios.yaml)
are of direct interest.
The 
[`Dockerfile`](https://github.com/Cecca/orthogonal-competition/blob/main/template/Dockerfile)
gives an example to build the `Dockerfile`.

The `competitors/faiss-hnsw` shows how to integrate the popular `faiss-hnsw` baseline
in this competition.

Running the competition
-----------------------

Assuming datasets are in the `datasets` directory, for a team `team` with a docker image
`team-image` we can the benchmark with the following command:

```
python evaluator.py evaluate --team team --image team-image \
                             --dataset data/data-file.hdf5
```

The evaluator:
  1. Extracts scenarios.yaml from the image (without running it) to discover
     scenario names.
  2. Spawns one fresh container per scenario, passing SCENARIO_NAME so the
     harness runs exactly that configuration.
  3. While the container runs, polls Docker's memory stats in a background
     thread to record the true peak RSS (cgroup-based, includes all C heap).
  4. Reads the flat results.hdf5 written by the harness and computes metrics.
  5. Stores one DB row per scenario.


Running on AWS
--------------

By default the containers run on the local Docker daemon. To instead run them on
an EC2 instance of a chosen type, add a `[runner]` block to the config (see the
commented example in `config.toml`) and run with the config-driven command:

```
python evaluator.py run --config config.toml --backend aws
```

With the AWS backend the evaluator provisions **one** EC2 instance of the
configured `instance_type`, runs every (team × dataset × scenario) container on
it, and terminates it when finished. Specifically it:
  1. launches the instance (boto3) with an existing key pair and security group;
  2. downloads the config's datasets onto the host over HTTP from
     `datasets_base_url` (default: the organizer's self-hosted directory);
  3. ships each locally-built competitor image to the host with `docker save` /
     `docker load` over SSH;
  4. runs the containers on the remote daemon (reached via an SSH-forwarded
     Docker socket) and copies each `results.hdf5` back — the measurement logic
     is identical to the local backend, so numbers are comparable;
  5. terminates the instance (set `terminate_after = false` to keep it).

Prerequisites: valid AWS credentials in the environment (e.g. `AWS_PROFILE`), an
existing EC2 **key pair** whose private key is at `key_path`, a **security group**
allowing inbound SSH from wherever you run the evaluator, and an Ubuntu **AMI**
(Docker and socat are installed by user-data if absent). The `--backend` flag
overrides whatever the config specifies.

### Setting up an AWS profile

The runner authenticates through `boto3`, which reads a named **profile** from
`~/.aws/credentials` and `~/.aws/config`. To create one:

1. In the AWS console, go to **IAM → Users → your user → Security credentials →
   Create access key**, choose the *Command Line Interface (CLI)* use case, and
   copy the **Access Key ID** and **Secret Access Key** (the secret is shown
   once). The identity needs EC2 permissions — `AmazonEC2FullAccess` is the
   simplest managed policy for testing.

2. Add the keys to `~/.aws/credentials`:

   ```ini
   [orthogonal]
   aws_access_key_id = AKIA...
   aws_secret_access_key = ...
   ```

   and the default region to `~/.aws/config`:

   ```ini
   [profile orthogonal]
   region = eu-west-1
   ```

   (Or run `aws configure --profile orthogonal` if you have the AWS CLI — it
   writes the same two files. The CLI is not required; `boto3` reads the files
   directly.)

3. Select the profile before running the evaluator:

   ```fish
   set -x AWS_PROFILE orthogonal        # fish
   # export AWS_PROFILE=orthogonal      # bash/zsh
   ```

4. Verify it authenticates:

   ```
   python -c "import boto3; print(boto3.Session().client('sts').get_caller_identity()['Arn'])"
   ```

   An `arn:aws:iam::...` line means the runner can reach AWS. If your
   organization uses **AWS SSO / Identity Center** instead of IAM access keys,
   configure the profile with `aws sso login` and set `AWS_PROFILE` the same way.


Championships
--------

Submissions are ranked in the following championships:

- _Sherlock Holmes_: the fastest approach achieving average recall above 0.95 wins
- _Bianconiglio_: the fastest approach achieving average recall above 0.8 wins
- _Dory_: the approach using the least amount of memory while achieving recall above 0.95,
  while at the same time not taking more than twice the time of the
  [`faiss-hnsw`](https://github.com/Cecca/orthogonal-competition/tree/main/competitors/faiss-hnsw)
  baseline with parameters `efConstruction: 100, M: 16, ef: 50`.
- _Marie Kondo_: the fastest approach at building the index, in a configuration that achieves at least 0.95 recall
- _Paperone_: the approache making the fewest distance computations, with an average recall at least 0.95.
  For the purpose of this championship, only _full Euclidean distance computations_ count.
  That is, sketches and product quantization (and similar techniques) _do not_ count towards the distance count.

  
Scenarios
---------

The submissions will be run in different scenarios:

- `high_recall`: submissions should aim for a recall of at least 0.95. This scenario
  is used for the _Sherlock Holmes_, _Marie Kondo_, and _Paperone_ prizes;
- `fast`: submissions should aim for a recall of at least 0.8. This scenario determines
  the _Bianconiglio_ prize;
- `memory`: submissions should aim for a recall of at least 0.95. This scenario
  is used for the _Dory_ prize.




