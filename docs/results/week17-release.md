# Week 17 release preparation

The v1.0.0 candidate includes the Python library and optional Studio, model serving, and
routing tools. On November 18, 2026, the public listings showed v0.1.0 on GitHub and the
v0.0.1 placeholder on PyPI. The paired Week 14 live study and Week 13 live routing
comparison remain pending.
[GitHub releases](https://github.com/krtk06/dataquorum/releases) ·
[PyPI package](https://pypi.org/project/dataquorum/)

## Release materials

- [README and quick start](../../README.md)
- [Captioned Studio demo](../media/studio-demo.mp4)
- [v1.0.0 changelog](../../CHANGELOG.md)
- [Detecting false successes in data-science agents](../writeups/verified-agent-outputs.md)
- [What more training examples changed in agent routing](../writeups/learning-to-route.md)

The 46-second video starts with the offline sample, runs three steps, opens a step's verification
checks, shows the verified revenue chart, and saves the pipeline JSON. The JSON describes
the steps, inputs, and requests. The Python export uses the same recipe with a local input
and the configured live agent runtime.

## Local verification

The wheel installed in an empty Python 3.12 environment. The base package imported all
eight agent specifications. Installing the Studio extra enabled the CLI, and the browser
walkthrough ran from the installed wheel in a separate working directory. Docker access
and API credentials were disabled there. The three steps verified, the expected revenue
chart rendered, and the downloaded JSON contained the expected three-step recipe.

The final wheel contains the same 103 package files as the wheel used in that walkthrough.
The source archive independently built and installed in a separate Python 3.12 environment.
Both artifacts passed `twine check --strict` and matched their SHA-256 checksums. The
repository suite passed all 396 tests, including Docker checks. Ruff, strict mypy, and
the uv lock check passed. The suite emitted an MLflow dependency deprecation warning.

## Build and verify locally

1. In PowerShell, from the repository root, build both distributions.

   ```powershell
   uv lock --check
   uv build --no-sources --out-dir dist/v1.0.0
   ```

2. Check that the wheel and source archive match the digests in
   `dist/v1.0.0/SHA256SUMS.txt` from the prepared build.

   ```powershell
   Get-FileHash -Algorithm SHA256 dist/v1.0.0/dataquorum-1.0.0*
   ```

3. In a fresh Python 3.12 environment, install the wheel with the Studio extra and launch it.
   Select **Run pipeline** and check the verified steps and chart. The demo works without
   Docker or API credentials.

   ```powershell
   uv venv --python 3.12 runs/release-check
   $python = 'runs/release-check/Scripts/python.exe'
   $wheel = (Resolve-Path 'dist/v1.0.0/dataquorum-1.0.0-py3-none-any.whl').Path
   uv pip install --python $python "dataquorum[studio] @ $wheel"
   uv pip check --python $python
   & 'runs/release-check/Scripts/dataquorum-studio.exe' --port 8503
   ```

## Publish when approved

1. Review the release changes and confirm the intended commit and artifacts.
2. For the GitHub release, tag the approved commit as `v1.0.0` and attach the wheel,
   source archive, and checksums from `dist/v1.0.0`.
3. For PyPI, the project owner runs the following in PowerShell 7 from the repository root.
   Enter a PyPI token only at the masked prompt.

   ```powershell
   $env:UV_PUBLISH_TOKEN = Read-Host 'PyPI API token' -MaskInput
   try {
       uv publish dist/v1.0.0/dataquorum-1.0.0-py3-none-any.whl dist/v1.0.0/dataquorum-1.0.0.tar.gz
       if ($LASTEXITCODE -ne 0) { throw 'uv publish failed' }
   } finally {
       Remove-Item Env:UV_PUBLISH_TOKEN -ErrorAction SilentlyContinue
   }
   ```
