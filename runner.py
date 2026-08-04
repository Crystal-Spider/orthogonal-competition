#!/usr/bin/env python3
"""
Execution backends for the NNS competition evaluator.
=====================================================

The evaluator talks to Docker exclusively through the Python SDK, so *where*
the containers run is just a matter of which ``docker.DockerClient`` it is
handed.  This module hides that behind a small ``Runner`` abstraction:

- ``LocalRunner``  – the historical behaviour: run on the local daemon.
- ``AwsRunner``    – provision one EC2 instance (of a chosen instance type),
                     run every container on it over an SSH-tunnelled Docker
                     socket, then terminate it.

Both expose the same tiny interface used by ``evaluator.py``:

    with make_runner(cfg) as runner:
        runner.ensure_image(image)                 # make image available
        client   = runner.client                   # a docker.DockerClient
        data_dir = runner.host_data_dir(local_dir) # host path to bind-mount ro
        rc, out  = runner.host_exec("nvidia-smi")  # shell out on the container host

``host_exec`` is what lets the evaluator measure resources where the containers
actually run: with the AWS backend that is the EC2 instance, not this machine.

``make_runner`` selects the backend from ``cfg["runner"]["backend"]`` and
defaults to ``local`` so existing configs keep working unchanged.
"""

from __future__ import annotations

import io
import logging
import select
import socket
import subprocess
import threading
import time
from pathlib import Path

import docker

log = logging.getLogger(__name__)

DEFAULT_DATASETS_BASE_URL = "https://www.dei.unipd.it/~ceccarello/orthogonal-datasets"
REMOTE_DATA_DIR = "/data"            # where datasets are staged on the EC2 host

# Command that bridges the host's Docker socket to stdin/stdout.  Shipped with
# the Docker CLI itself (>= 18.09), so nothing extra has to be installed, and
# `sudo` sidesteps the root-owned socket's permissions.
REMOTE_DOCKER_DIAL = "sudo docker system dial-stdio"


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------

class LocalRunner:
    """Run containers on the local Docker daemon (default backend)."""

    backend = "local"

    def __init__(self):
        self.client = docker.from_env()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.client.close()
        except Exception:
            pass
        return False

    def host_data_dir(self, local_data_dir) -> str:
        # The daemon shares our filesystem, so the dataset's own directory is
        # exactly the host path to bind-mount.
        return str(Path(local_data_dir).resolve())

    def ensure_image(self, image: str) -> None:
        # Images built locally are already visible to the local daemon.
        return None

    def host_exec(self, command: str, timeout: float = 15.0) -> tuple[int, str]:
        """Run a shell command on the container host — here, this machine."""
        try:
            proc = subprocess.run(
                ["/bin/sh", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
            return proc.returncode, proc.stdout
        except Exception:
            return 1, ""


# ---------------------------------------------------------------------------
# TCP <-> SSH channel forwarder (docker system dial-stdio)
# ---------------------------------------------------------------------------

class _PortForward:
    """
    Accepts connections on a local ephemeral TCP port and pipes each of them,
    over its own paramiko exec channel, into ``docker system dial-stdio`` on the
    remote host — which in turn talks to the host's Docker socket.  The Docker
    SDK then reaches the remote daemon through a plain ``tcp://127.0.0.1:<port>``
    URL, with no daemon reconfiguration, extra packages or listening port on the
    host side.
    """

    def __init__(self, transport, remote_command: str):
        self._transport = transport
        self._remote_command = remote_command
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(16)
        self._server.settimeout(1.0)
        self.port = self._server.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                sock, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(sock,), daemon=True).start()

    def _handle(self, sock: socket.socket):
        try:
            chan = self._transport.open_session()
            chan.exec_command(self._remote_command)
        except Exception as exc:
            log.debug("forward: channel open failed: %s", exc)
            sock.close()
            return
        try:
            while True:
                r, _, _ = select.select([sock, chan], [], [], 1.0)
                if sock in r:
                    data = sock.recv(65536)
                    if not data:
                        break
                    chan.sendall(data)
                if chan in r:
                    # stderr carries the remote command's own diagnostics (a
                    # missing binary, a sudo refusal); surface them instead of
                    # letting the SDK report a bare connection reset.
                    if chan.recv_stderr_ready():
                        err = chan.recv_stderr(65536).decode("utf-8", "replace").strip()
                        if err:
                            log.error("[host] %s: %s", self._remote_command, err)
                    if chan.recv_ready():
                        data = chan.recv(65536)
                        if not data:
                            break
                        sock.sendall(data)
                    elif chan.eof_received or chan.exit_status_ready():
                        break
        except Exception:
            pass
        finally:
            try:
                chan.close()
            finally:
                sock.close()

    def stop(self):
        self._stop.set()
        try:
            self._server.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AWS backend
# ---------------------------------------------------------------------------

# Cloud-init run at boot: make sure Docker is present and running.  Harmless if
# the AMI already ships it, which is the common case.
_DEFAULT_USER_DATA = """#!/bin/bash
set -x
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sh
fi
usermod -aG docker {ssh_user} || true
systemctl enable --now docker || true
if ! command -v curl >/dev/null 2>&1; then   # datasets are staged with curl
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y && apt-get install -y curl
fi
"""


class AwsRunner:
    """
    Provision a single EC2 instance, run all containers on it, then terminate.

    Configuration (the ``[runner]`` block, already validated by
    ``evaluator.load_config``) must provide at least ``instance_type``,
    ``region``, ``ami``, ``key_name`` and ``key_path``.  ``security_group_ids``
    must allow inbound SSH from wherever the evaluator runs.
    """

    backend = "aws"

    def __init__(self, cfg: dict, dataset_filenames):
        self.cfg = cfg
        self.dataset_filenames = list(dataset_filenames)
        self.ssh_user  = cfg.get("ssh_user", "ubuntu")
        self.base_url  = cfg.get("datasets_base_url", DEFAULT_DATASETS_BASE_URL).rstrip("/")
        self.terminate_after = bool(cfg.get("terminate_after", True))

        self._local = docker.from_env()   # source daemon for image save/load
        self._ec2 = None
        self._instance_id = None
        self._ssh = None
        self._forward = None
        self._loaded = set()              # images already transferred to the host
        self.client = None                # remote docker.DockerClient (tcp://)
        # host_exec is called from the resource-monitor thread while the main
        # thread may also be driving the SSH session; serialise the two.
        self._ssh_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        import boto3  # imported lazily so the local backend needs no boto3

        self._ec2 = boto3.client("ec2", region_name=self.cfg["region"])
        try:
            self._launch_instance()
            host = self._wait_running()
            self._connect_ssh(host)
            self._wait_docker_ready()
            self._stage_datasets()
            self._open_docker_client()
        except Exception:
            # Never leak a running instance if setup fails partway through.
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        if self._forward is not None:
            self._forward.stop()
            self._forward = None
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
        if self._instance_id and self.terminate_after:
            log.info("Terminating instance %s", self._instance_id)
            try:
                self._ec2.terminate_instances(InstanceIds=[self._instance_id])
            except Exception as exc:
                log.error("Failed to terminate %s: %s", self._instance_id, exc)
        elif self._instance_id:
            log.warning("terminate_after=false: instance %s left running", self._instance_id)
        try:
            self._local.close()
        except Exception:
            pass
        return False

    # -- provisioning ------------------------------------------------------

    def _launch_instance(self):
        cfg = self.cfg
        params = dict(
            ImageId=cfg["ami"],
            InstanceType=cfg["instance_type"],
            KeyName=cfg["key_name"],
            MinCount=1,
            MaxCount=1,
            UserData=cfg.get("user_data", _DEFAULT_USER_DATA).format(ssh_user=self.ssh_user),
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "orthogonal-competition-runner"}],
            }],
        )
        if cfg.get("security_group_ids"):
            params["SecurityGroupIds"] = list(cfg["security_group_ids"])
        if cfg.get("iam_instance_profile"):
            params["IamInstanceProfile"] = {"Name": cfg["iam_instance_profile"]}
        vol_gb = int(cfg.get("volume_size_gb", 100))
        params["BlockDeviceMappings"] = [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": vol_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
        }]

        log.info("Launching %s in %s (ami=%s) ...", cfg["instance_type"], cfg["region"], cfg["ami"])
        resp = self._ec2.run_instances(**params)
        self._instance_id = resp["Instances"][0]["InstanceId"]
        log.info("Instance %s requested", self._instance_id)

    def _wait_running(self) -> str:
        log.info("Waiting for %s to reach 'running' ...", self._instance_id)
        self._ec2.get_waiter("instance_running").wait(InstanceIds=[self._instance_id])
        desc = self._ec2.describe_instances(InstanceIds=[self._instance_id])
        inst = desc["Reservations"][0]["Instances"][0]
        host = inst.get("PublicDnsName") or inst.get("PublicIpAddress")
        if not host:
            raise RuntimeError(
                f"Instance {self._instance_id} has no public address; check the "
                "subnet's auto-assign public IP setting."
            )
        log.info("Instance %s running at %s", self._instance_id, host)
        return host

    def _connect_ssh(self, host: str, timeout: int = 240):
        import paramiko

        key_path = str(Path(self.cfg["key_path"]).expanduser())
        pkey = self._load_key(key_path)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        log.info("Waiting for SSH on %s ...", host)
        deadline = time.monotonic() + timeout
        last_exc = None
        while time.monotonic() < deadline:
            try:
                client.connect(
                    hostname=host, username=self.ssh_user, pkey=pkey,
                    timeout=15, banner_timeout=15, auth_timeout=15,
                )
                self._ssh = client
                log.info("SSH established")
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(5)
        raise RuntimeError(f"SSH to {host} did not become ready: {last_exc}")

    @staticmethod
    def _load_key(key_path: str):
        import paramiko
        for loader in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
            try:
                return loader.from_private_key_file(key_path)
            except Exception:
                continue
        raise RuntimeError(f"Could not load private key {key_path} (unsupported type or passphrase-protected)")

    def _ssh_exec(self, command: str, check: bool = True, log_output: bool = False) -> int:
        with self._ssh_lock:
            stdin, stdout, stderr = self._ssh.exec_command(command)
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            rc = stdout.channel.recv_exit_status()
        if log_output and out.strip():
            log.info("[host] %s", out.strip())
        if rc != 0:
            log.error("[host] command failed (rc=%d): %s\n%s", rc, command, err.strip()[-2000:])
            if check:
                raise RuntimeError(f"Remote command failed (rc={rc}): {command}")
        return rc

    def _wait_docker_ready(self, timeout: int = 420):
        """User-data may still be installing Docker; wait for it to answer."""
        log.info("Waiting for Docker on the host ...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ssh_exec("sudo docker info >/dev/null 2>&1", check=False) == 0:
                log.info("Docker is up on the host")
                return
            time.sleep(5)
        raise RuntimeError("Docker did not become ready on the host in time")

    def _stage_datasets(self):
        self._ssh_exec(f"sudo mkdir -p {REMOTE_DATA_DIR} && sudo chown $(whoami) {REMOTE_DATA_DIR}")
        for name in self.dataset_filenames:
            url = f"{self.base_url}/{name}"
            dest = f"{REMOTE_DATA_DIR}/{name}"
            log.info("Downloading dataset %s ...", name)
            # -C - resumes a partial file; --retry rides out transient hiccups.
            self._ssh_exec(
                f"curl -fL --retry 3 -C - -o {dest!r} {url!r} || curl -fL --retry 3 -o {dest!r} {url!r}"
            )
        log.info("Datasets staged in %s", REMOTE_DATA_DIR)

    def _open_docker_client(self, timeout: int = 60):
        # Fail loudly here rather than through a stream of refused channels: the
        # forward is only as good as this command.
        if self._ssh_exec(f"{REMOTE_DOCKER_DIAL} </dev/null >/dev/null", check=False) != 0:
            raise RuntimeError(
                f"{REMOTE_DOCKER_DIAL!r} does not work on the host; the Docker CLI "
                "must be >= 18.09 and the SSH user must have passwordless sudo."
            )
        self._forward = _PortForward(self._ssh.get_transport(), REMOTE_DOCKER_DIAL).start()
        base_url = f"tcp://127.0.0.1:{self._forward.port}"
        log.info("Connecting Docker SDK via %s -> host %s", base_url, REMOTE_DOCKER_DIAL)
        deadline = time.monotonic() + timeout
        last_exc = None
        while time.monotonic() < deadline:
            try:
                client = docker.DockerClient(base_url=base_url, timeout=120)
                client.ping()
                self.client = client
                log.info("Remote Docker reachable")
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(2)
        raise RuntimeError(f"Could not reach remote Docker over the forward: {last_exc}")

    # -- interface used by the evaluator -----------------------------------

    def host_data_dir(self, local_data_dir) -> str:
        # All datasets were downloaded flat into REMOTE_DATA_DIR by basename.
        return REMOTE_DATA_DIR

    def host_exec(self, command: str, timeout: float = 15.0) -> tuple[int, str]:
        """
        Run a shell command on the container host — here, the EC2 instance.

        Never raises: callers poll with it in a background thread, where a
        transient SSH hiccup must not take the run down.  A failure is reported
        as a non-zero return code with empty output.
        """
        ssh = self._ssh
        if ssh is None:
            return 1, ""
        try:
            with self._ssh_lock:
                _, stdout, _ = ssh.exec_command(command, timeout=timeout)
                out = stdout.read().decode("utf-8", "replace")
                rc = stdout.channel.recv_exit_status()
            return rc, out
        except Exception:
            return 1, ""

    def ensure_image(self, image: str) -> None:
        """Copy a locally-built image to the remote daemon (docker save|load)."""
        if image in self._loaded:
            return
        log.info("Transferring image %s to the host ...", image)
        try:
            local_image = self._local.images.get(image)
        except docker.errors.ImageNotFound:
            raise RuntimeError(f"Image {image!r} not found on the local daemon; build it first.")
        buf = io.BytesIO(b"".join(local_image.save(named=True)))
        buf.seek(0)
        self.client.images.load(buf.read())
        self._loaded.add(image)
        log.info("Image %s loaded on the host", image)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_runner(cfg: dict):
    """
    Build the runner selected by ``cfg["runner"]["backend"]`` (default local).

    ``cfg`` is the full config dict returned by ``evaluator.load_config`` – it
    carries both the ``runner`` block and the ``datasets`` list (whose basenames
    the AWS backend downloads onto the host).
    """
    runner_cfg = cfg.get("runner") or {}
    backend = runner_cfg.get("backend", "local")
    if backend == "local":
        return LocalRunner()
    if backend == "aws":
        dataset_filenames = [Path(d).name for d in cfg["datasets"]]
        return AwsRunner(runner_cfg, dataset_filenames)
    raise ValueError(f"Unknown runner backend {backend!r}; expected 'local' or 'aws'.")
