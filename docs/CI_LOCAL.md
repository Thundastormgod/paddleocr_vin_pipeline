# Local CI — the gate without GitHub Actions

The GitHub Actions billing lock on this account blocks every hosted
runner job. The gate itself does not depend on GitHub: `scripts/local_ci.py`
executes the same jobs with the same commands natively.

## Usage

```bash
make gate          # full runnable set (~4 min): secrets, lint, typecheck,
                   # pip-audit, test (coverage >= 30%), build
make gate-quick    # fast subset (~2 min): secrets, lint, test
python scripts/local_ci.py --job lint     # any single job
python scripts/local_ci.py --bootstrap    # one-time install of runner tools
```

`make hooks` activates a pre-push hook that runs the quick subset on every
push containing Python/config changes; a red gate refuses the push
(`git push --no-verify` bypasses).

## Job coverage vs .github/workflows/ci.yml

| ci.yml job | local runner | notes |
|---|---|---|
| secrets-scan | yes | identical git checks |
| lint | yes | blocking critical subset + advisory full report |
| typecheck | yes* | needs `.github/mypy-baseline.txt` (present on this branch; on older trees it reports SKIP and passes) |
| pip-audit | yes | audits `.github/requirements-ci.txt` pins |
| test | yes | single py3.12 leg locally; CI's 3.10/3.11 legs need Actions or act |
| build | yes | sdist/wheel + console-script resolver |
| docker / model-gate / codeql | no | container runtime / repo secrets / CodeQL service required |

## Container-faithful alternative (`act`)

To execute the workflow YAML files verbatim in containers:

```bash
brew install colima docker act && colima start
act -P ubuntu-latest=catthehacker/ubuntu:act-latest pull_request \
    -W .github/workflows/ci.yml --job test
```

Caveats verified against action implementations: `upload-artifact`,
`attest-build-provenance`, and `codeql-action` require GitHub services and
will fail under act; everything else in ci.yml runs. The native runner is
the supported path on this machine.
