# Third-Party Notices

This file summarizes third-party software and model components used by the
CPU production image. The project itself is licensed under the MIT License; see
`LICENSE.txt`.

Full runtime package metadata and license identifiers can be regenerated with:

```bash
python scripts/audit_licenses.py --write THIRD_PARTY_LICENSES.md
```

The CPU image also generates `/app/THIRD_PARTY_LICENSES.md` during the Docker
build.

## Speech recognition models

Whisper-family models are downloaded at runtime and are not bundled in this
image. The supported model repositories and their declared licenses are:

| Model | License | License URL |
| -- | -- | -- |
| `distil-whisper/distil-large-v3.5` | MIT | https://huggingface.co/distil-whisper/distil-large-v3.5 |
| `distil-whisper/distil-large-v3` | MIT | https://huggingface.co/distil-whisper/distil-large-v3 |
| `openai/whisper-large-v3` | Apache-2.0 | https://huggingface.co/openai/whisper-large-v3 |
| `openai/whisper-large-v3-turbo` | MIT | https://huggingface.co/openai/whisper-large-v3-turbo |

## Speaker diarization model

The default diarization pipeline is:

- **Model:** `pyannote/speaker-diarization-community-1`
- **Copyright:** pyannoteAI and contributors
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Model page:** https://huggingface.co/pyannote/speaker-diarization-community-1
- **License text:** https://creativecommons.org/licenses/by/4.0/legalcode

The model repository contains the segmentation, speaker-embedding, and PLDA
weights used by this pipeline. The model is downloaded at runtime and is not
modified by this project.

Attribution notice:

> This product uses the `pyannote/speaker-diarization-community-1` speaker
> diarization model by pyannoteAI, licensed under CC-BY-4.0. The model weights
> are unmodified. Source: https://huggingface.co/pyannote/speaker-diarization-community-1

The model is gated on Hugging Face. Operators must accept the model terms before
the model can be downloaded.

## Noise reduction model

The GPU and CPU-capable noise-reduction path uses:

- **Model:** `dpdfnet8.onnx` from the DPDFNet project
- **Copyright:** DPDFNet contributors / CEVA
- **License:** Apache-2.0
- **Project:** https://github.com/ceva-ip/DPDFNet
- **Model card:** https://huggingface.co/Ceva-IP/DPDFNet
- **Model source:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

The GPU image bundles the unmodified `dpdfnet8.onnx` model and its Apache-2.0
license at `/app/models/DPDFNet-LICENSE`.

DPDFNet is a speech-enhancement model. It reduces background noise but does not
perform vocal/instrument stem separation.

## Python runtime packages

The runtime includes MIT, BSD, Apache-2.0, ISC, and PSF-licensed Python
packages. Principal components include:

- **PyTorch / torchaudio** — BSD-3-Clause
- **Transformers, huggingface_hub, safetensors, tokenizers, Gradio, Lightning,
  torchmetrics** — Apache-2.0
- **pyannote.audio, pyannote.core, pyannote.database, pyannote.metrics,
  pyannote.pipeline, pyannoteai-sdk** — MIT
- **sherpa-onnx, sherpa-onnx-core** — Apache-2.0
- **FastAPI, pydantic, pydub, stable-ts, torch-audiomentations,
  asteroid-filterbanks** — MIT
- **NumPy, SciPy, pandas, scikit-learn, uvicorn, starlette** — BSD
- **FFmpeg Python bindings and system FFmpeg** — LGPL/GPL terms as shipped by
  the Debian package; see the package copyright file for details.
- **libsndfile** — LGPL terms as shipped by the Debian package.

## PyTorch and Intel MKL notice

The official PyTorch CPU wheels statically link Intel Math Kernel Library (MKL)
and oneDNN components. PyTorch is distributed under the BSD-3-Clause license.
Intel MKL is redistributable under the Intel Simplified Software License (ISSL)
but is not an OSI-approved license. The Intel MKL license notice should be
included with binary distributions. The ROCm PyTorch build does not use MKL; its
bundled AMD ROCm libraries are covered by the PyTorch `NOTICE` and their
upstream licenses. See:

- https://github.com/pytorch/pytorch/blob/main/LICENSE
- https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html

## Demucs

Demucs is intentionally excluded from the distributed builds. The Demucs source
code is MIT-licensed, but the upstream pretrained weights are provided for
scientific/research use only and are not covered by the MIT license. The
`--demucs` compatibility flag now selects Apache-2.0 DPDFNet noise reduction
instead. The image does not install the `demucs` package and cannot download or
run those weights.

## License texts

- MIT: https://opensource.org/license/mit
- BSD-3-Clause: https://opensource.org/license/bsd-3-clause
- Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
- CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/legalcode
- LGPL-2.1: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html
- GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html

For source-offer obligations covering LGPL/GPL components shipped by the base
image, retain the corresponding Debian source package information for the
image build.
