# pip Package and PyPI Release

The distribution name is `mpt`; the project name remains MoneyPrinterTurbo, and the version pending release is `1.3.9`.
`1.3.8` was published to [PyPI](https://pypi.org/project/mpt/1.3.8/) on 2026-10-16 under the
`harry0703` account, with both a wheel and an sdist.
Published files cannot be overwritten; later code changes require bumping the version, then rebuilding and re-verifying.

## Building and Installing

Python 3.11 or newer is required. Dependencies are resolved and installed along with the wheel. FFmpeg
prefers the system installation, but the binary provided by imageio-ffmpeg can also be used.

Install from PyPI and start as a user:

```sh
python -m pip install mpt
mpt webui
mpt cli --help
```

Build and verify locally:

```sh
uv build --out-dir storage/package/mpt-dist
python -m pip install ./storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl
mpt --help
mpt webui
mpt cli --help
```

Before an official upload, confirm the PyPI account, the target version and the distribution hashes, and avoid
bulk-uploading from a directory that contains stale packages. Every release requires explicit user authorization;
nothing is published automatically on commit or push.

## Pre-release Checks

```sh
uvx twine check --strict \
  storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl \
  storage/package/mpt-dist/mpt-1.3.9.tar.gz
```

Check the PyPI description page, the distribution metadata and dependency installability, and complete a video
generation run in a fresh virtual environment.
Perform the official release only after explicit authorization to upload; account passwords, verification codes
and API tokens must never be written into the repository, command examples or logs. The README in each language
provides the user installation entry point; the PyPI description is maintained separately in
`docs/pypi-description.md`, and updating repository docs does not automatically change the already-published PyPI page.

`mpt webui` listens on `127.0.0.1` only by default, picks a free port starting from 8501, and opens the browser automatically.
Use `--host` and `--port` to specify the address, and `--headless` to prevent the browser from opening:

```sh
mpt webui --port 8503 --headless
```

The CLI reuses the arguments, task pipeline, exit codes and JSON output of `python cli.py`. A complete offline example:

```sh
mpt cli --video-script "A short video from local materials." \
  --video-source local --video-materials /absolute/path/clip.mp4 \
  --custom-audio-file /absolute/path/narration.wav \
  --no-subtitle-enabled --bgm-type none
```

Online script generation, materials, TTS and similar features still require the corresponding service configuration,
keys and network access. Whisper models are loaded on demand and are not bundled in the wheel. CLI logs are written
to stderr; on success, stdout prints JSON that includes the video path.

## Configuration and Data

Running from source continues to use the project root directory. The installed package does not depend on the
current working directory; its default runtime directory is:

| Platform | Runtime directory |
| --- | --- |
| macOS | `~/Library/Application Support/MoneyPrinterTurbo` |
| Windows | `%LOCALAPPDATA%\MoneyPrinterTurbo` |
| Linux | `$XDG_DATA_HOME/MoneyPrinterTurbo`, defaulting to `~/.local/share/MoneyPrinterTurbo` |

The runtime directory can be set explicitly with `MPT_HOME` (also supported when running from source). Within it:

- `config.toml`: initialized on first run from a template that contains no keys; later startups never overwrite an existing configuration.
- `storage/tasks`: tasks and generated videos; other caches and uploaded files live under `storage`.
- `resource/fonts`, `resource/songs`: the built-in resources are copied on first use, and you can add your own files.
- `models`: manually downloaded Whisper models can be placed here.

User resources are only filled in where files are missing; upgrades never overwrite existing files. The package
includes two default Chinese fonts, a Vietnamese font and the built-in background music; the alternative
`STHeitiLight.ttc` and `MicrosoftYaHeiNormal.ttc` are excluded from the distribution to keep its size down.
If you need those weights, place them in `resource/fonts` under the runtime directory.
The resources and appearance of the source version are unchanged.

## Local Verification Requirements

Rebuild the wheel from the sdist, install it into a fresh virtual environment that does not inherit system packages,
leave the source directory and clear `PYTHONPATH`, then verify `mpt cli --help`, the actual WebUI pages and video
generation. The output video must decode completely with FFmpeg, and confirm that no configuration or artifacts were
written into site-packages. The distribution must not contain the local `config.toml`, keys, caches, task artifacts
or local collaboration instructions.
