"""Escape tests: hostile code must be contained by the sandbox (spec S5).

Each probe is the body of a `transform(df)` function. The body tries one escape and does
nothing else; if the body finishes, the escape worked. The probe returns one row saying
whether it was blocked, so a failure message shows exactly what got through.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from dataquorum.sandbox.docker_executor import DockerExecutor

pytestmark = pytest.mark.docker


def probe(body: str) -> str:
    return (
        "import os, socket, subprocess, sys\n"
        "import pandas as pd\n\n"
        "def transform(df):\n"
        "    try:\n" + textwrap.indent(textwrap.dedent(body).strip(), " " * 8) + "\n"
        "        return pd.DataFrame({'blocked': [False], 'detail': ['escape succeeded']})\n"
        "    except BaseException as exc:\n"
        "        return pd.DataFrame({'blocked': [True], 'detail': [f'{type(exc).__name__}: {exc}']})\n"
    )


ESCAPES = {
    "tcp_connection": "socket.create_connection(('1.1.1.1', 53), timeout=3)",
    "dns_lookup": "socket.getaddrinfo('example.com', 80)",
    "write_root_filesystem": "open('/etc/dq_escape', 'w').write('x')",
    "write_usr": "open('/usr/local/lib/dq_escape.py', 'w').write('x')",
    "write_input_mount": "open('/io/in/input.parquet', 'ab').write(b'x')",
    "tamper_with_runner": "open('/io/in/runner.py', 'a').write('# x')",
    "become_root": "os.setuid(0)",
    "chown_to_root": "os.chown('/tmp', 0, 0)",
    "raw_socket": "socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)",
    "mount_filesystem": "subprocess.run(['mount', '-t', 'tmpfs', 'none', '/mnt'], check=True, capture_output=True)",
    "su_to_root": "subprocess.run(['su', '-c', 'id'], check=True, capture_output=True, input=b'', timeout=5)",
    "docker_socket": "os.stat('/var/run/docker.sock')",
    "execute_from_tmp": """
        open('/tmp/x.sh', 'w').write('#!/bin/sh\\necho pwned\\n')
        os.chmod('/tmp/x.sh', 0o755)
        subprocess.run(['/tmp/x.sh'], check=True, capture_output=True)
    """,
    "kernel_sysrq": "open('/proc/sysrq-trigger', 'w').write('h')",
    "host_secret_in_env": "assert 'OPENCODE_GO_API_KEY' in os.environ, 'not visible'",
    # DQ_HOST_CANARY is set in the test process (see host_canary); Windows-only names prove
    # nothing from the host shell leaked. Image-defined variables (PATH, LANG...) are fine.
    "any_host_env": "assert any(k in os.environ for k in ('DQ_HOST_CANARY', 'USERPROFILE', 'USERNAME', 'COMPUTERNAME', 'APPDATA')), 'none visible'",
    "fill_tmp_disk": """
        with open('/tmp/fill.bin', 'wb') as f:
            for _ in range(400):
                f.write(b'0' * 1024 * 1024)
    """,
    "fork_many_processes": """
        procs = [subprocess.Popen(['sleep', '5']) for _ in range(200)]
    """,
    "allocate_too_much_memory": "x = bytearray(3 * 1024 ** 3)",
}  # fmt: skip


@pytest.fixture(autouse=True)
def host_canary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DQ_HOST_CANARY", "if-you-can-read-this-the-host-env-leaked")


@pytest.mark.parametrize("name", list(ESCAPES))
def test_escape_is_blocked(
    name: str, executor: DockerExecutor, input_path: Path, tmp_path: Path
) -> None:
    result = executor.run(probe(ESCAPES[name]), input_path, tmp_path / name)
    if not result.ok:  # the sandbox stopped the code outright (e.g. killed for memory)
        assert result.error
        return
    assert result.output_path is not None
    row = pd.read_parquet(result.output_path).iloc[0]
    assert row["blocked"], f"{name}: {row['detail']}"


def test_runs_as_unprivileged_user_with_no_capabilities(
    executor: DockerExecutor, input_path: Path, tmp_path: Path
) -> None:
    code = (
        "import os, pandas as pd\n"
        "def transform(df):\n"
        "    status = open('/proc/self/status').read().splitlines()\n"
        "    caps = {l.split(':')[0]: l.split(':')[1].strip() for l in status if l.startswith('Cap')}\n"
        "    return pd.DataFrame({'uid': [os.getuid()], 'gid': [os.getgid()],\n"
        "                         'cap_eff': [caps['CapEff']], 'cap_bnd': [caps['CapBnd']],\n"
        "                         'no_new_privs': [[l for l in status if l.startswith('NoNewPrivs')][0]]})\n"
    )
    result = executor.run(code, input_path, tmp_path)
    assert result.ok, result.error
    assert result.output_path is not None
    row = pd.read_parquet(result.output_path).iloc[0]
    assert (row["uid"], row["gid"]) == (10001, 10001)
    assert int(row["cap_eff"], 16) == 0 and int(row["cap_bnd"], 16) == 0
    assert row["no_new_privs"].split()[-1] == "1"


def test_only_expected_mounts_are_writable(
    executor: DockerExecutor, input_path: Path, tmp_path: Path
) -> None:
    code = (
        "import pandas as pd\n"
        "def transform(df):\n"
        "    rows = [l.split() for l in open('/proc/mounts')]\n"
        "    return pd.DataFrame({'target': [r[1] for r in rows], 'opts': [r[3] for r in rows]})\n"
    )
    result = executor.run(code, input_path, tmp_path)
    assert result.ok, result.error
    assert result.output_path is not None
    mounts = pd.read_parquet(result.output_path)
    writable = mounts[mounts["opts"].str.split(",").str[0] == "rw"]["target"].tolist()
    # /dev, /dev/shm-style kernel mounts may be rw; no host-backed path except /tmp (tmpfs).
    host_like = [t for t in writable if t.startswith(("/io", "/home", "/root", "/usr", "/etc"))]
    assert host_like == []
    assert mounts.loc[mounts["target"] == "/", "opts"].iloc[0].startswith("ro")
    assert mounts.loc[mounts["target"] == "/io/in", "opts"].iloc[0].startswith("ro")
    tmp_opts = mounts.loc[mounts["target"] == "/tmp", "opts"].iloc[0]
    assert "noexec" in tmp_opts and "nosuid" in tmp_opts
