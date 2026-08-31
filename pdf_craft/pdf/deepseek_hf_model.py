from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

from doc_page_extractor.check_env import check_env
from doc_page_extractor.extraction_context import ExtractionContext
from doc_page_extractor.injection import InferWithInterruption, preprocess_model
from doc_page_extractor.types import DeepSeekOCRSize
from huggingface_hub import snapshot_download
from readerwriterlock import rwlock
from transformers import AutoModel, AutoTokenizer

from .types import DeepSeekOCRModelId

_REPO_IDS: dict[DeepSeekOCRModelId, str] = {
    "deepseek-ocr": "deepseek-ai/DeepSeek-OCR",
    "deepseek-ocr-2": "deepseek-ai/DeepSeek-OCR-2",
}


def huggingface_cache_model_dir(model_id: DeepSeekOCRModelId) -> str:
    repo_id = _REPO_IDS[model_id]
    return "models--" + repo_id.replace("/", "--")


def _is_model_snapshot(path: Path) -> bool:
    return (path / "config.json").is_file()


def _latest_snapshot_under(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.is_dir():
        return None
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return None
    return max(snapshot_dirs, key=lambda d: d.stat().st_mtime)


def _snapshot_from_hub_cache(cache_model_dir: Path) -> str | None:
    ref_file = cache_model_dir / "refs" / "main"
    if ref_file.is_file():
        snapshot_hash = ref_file.read_text(encoding="utf-8").strip()
        snapshot_path = cache_model_dir / "snapshots" / snapshot_hash
        if snapshot_path.is_dir():
            return str(snapshot_path)

    latest = _latest_snapshot_under(cache_model_dir / "snapshots")
    if latest is None:
        return None
    return str(latest)


def resolve_local_pretrained_path(
    model_path: Path,
    cache_model_dir_name: str,
) -> str | None:
    root = model_path
    if _is_model_snapshot(root):
        return str(root)

    hub_cache_candidates: list[Path] = []
    for base in (root, root.parent):
        if base.is_dir():
            hub_cache_candidates.append(base / cache_model_dir_name)
    if root.name == cache_model_dir_name:
        hub_cache_candidates.insert(0, root)

    seen: set[Path] = set()
    for cache_model_dir in hub_cache_candidates:
        resolved = cache_model_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_dir():
            continue
        found = _snapshot_from_hub_cache(resolved)
        if found is not None and _is_model_snapshot(Path(found)):
            return found

    # Layout: {root}/snapshots/{hash}/ (no models-- wrapper)
    latest = _latest_snapshot_under(root / "snapshots")
    if latest is not None and _is_model_snapshot(latest):
        return str(latest)

    # Layout: hub cache metadata lives directly under {root}/
    found = _snapshot_from_hub_cache(root)
    if found is not None and _is_model_snapshot(Path(found)):
        return found

    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("models--"):
                found = _snapshot_from_hub_cache(child)
                if found is not None and _is_model_snapshot(Path(found)):
                    return found
            if _is_model_snapshot(child):
                return str(child)
            latest = _latest_snapshot_under(child / "snapshots")
            if latest is not None and _is_model_snapshot(latest):
                return str(latest)

    return None


def format_local_model_not_found_message(
    model_path: Path,
    cache_model_dir_name: str,
    repo_id: str,
) -> str:
    lines = [
        f"Local model not found under {model_path}.",
        f"Expected Hugging Face repo {repo_id!r} at one of:",
        f"  • {model_path}/{cache_model_dir_name}/snapshots/<hash>/",
        f"  • {model_path}/snapshots/<hash>/",
        f"  • {model_path}/ (flat download with config.json)",
        f"  • {model_path.parent}/{cache_model_dir_name}/snapshots/<hash>/",
        "",
    ]
    if not model_path.exists():
        lines.append("The models-cache-path does not exist.")
        return "\n".join(lines)

    lines.append("Contents of models-cache-path:")
    try:
        entries = sorted(model_path.iterdir(), key=lambda p: p.name)
    except OSError as error:
        lines.append(f"  (cannot list directory: {error})")
        entries = []

    if not entries:
        lines.append("  (empty directory)")
    else:
        for entry in entries[:25]:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"  {entry.name}{suffix}")
        if len(entries) > 25:
            lines.append(f"  ... and {len(entries) - 25} more")

    parent = model_path.parent
    sibling = parent / cache_model_dir_name
    if sibling.is_dir() and sibling != model_path:
        lines.append("")
        lines.append(f"Try --models-cache-path {parent}  (hub cache found at {sibling})")

    v1_name = "models--deepseek-ai--DeepSeek-OCR"
    v2_name = "models--deepseek-ai--DeepSeek-OCR-2"
    for base in (model_path, parent):
        if (base / v1_name).is_dir() and cache_model_dir_name == v2_name:
            lines.append("")
            lines.append(
                f"Found DeepSeek-OCR (v1) under {base / v1_name}, "
                "but OCR-2 weights are a separate download (DeepSeek-OCR-2)."
            )
            break

    config_files = list(model_path.glob("**/config.json"))
    if config_files:
        lines.append("")
        lines.append("config.json files found (may be wrong model or incomplete snapshot):")
        for config_path in config_files[:5]:
            lines.append(f"  {config_path}")

    return "\n".join(lines)

@dataclass
class _SizeConfig:
    base_size: int
    image_size: int
    crop_mode: bool


_SIZE_CONFIGS: dict[DeepSeekOCRSize, _SizeConfig] = {
    "tiny": _SizeConfig(base_size=512, image_size=512, crop_mode=False),
    "small": _SizeConfig(base_size=640, image_size=640, crop_mode=False),
    "base": _SizeConfig(base_size=1024, image_size=1024, crop_mode=False),
    "large": _SizeConfig(base_size=1280, image_size=1280, crop_mode=False),
    "gundam": _SizeConfig(base_size=1024, image_size=640, crop_mode=True),
}

_ATTN_IMPLEMENTATION: str
if find_spec("flash_attn") is not None:
    _ATTN_IMPLEMENTATION = "flash_attention_2"
else:
    _ATTN_IMPLEMENTATION = "eager"


@dataclass
class _Models:
    tokenizer: AutoTokenizer
    llms: list[AutoModel]


class DeepSeekOCRHuggingfaceModel:
    """Hugging Face backend for DeepSeek-OCR and DeepSeek-OCR-2."""

    def __init__(
        self,
        model_id: DeepSeekOCRModelId,
        model_path: Path | None,
        local_only: bool,
        enable_devices_numbers: Iterable[int] | None,
    ) -> None:
        if local_only and model_path is None:
            raise ValueError("model_path must be provided when local_only is True")

        self._model_id = model_id
        self._model_name = _REPO_IDS[model_id]
        self._cache_model_dir = huggingface_cache_model_dir(model_id)
        self._rwlock = rwlock.RWLockFair()
        self._model_path: Path | None = model_path
        self._local_only = local_only
        self._models: _Models | None = None
        self._enable_devices_numbers: Iterable[int] | None = enable_devices_numbers
        self._device_number_to_index: list[int | None] | None = None

    def download(self, revision: str | None) -> None:
        with self._rwlock.gen_wlock():
            snapshot_download(
                repo_id=self._model_name,
                repo_type="model",
                revision=revision,
                force_download=True,
                cache_dir=self._cache_dir(),
            )
            if self._model_path is not None and self._find_pretrained_path() is None:
                raise RuntimeError(
                    f"Model downloaded but not found in expected cache structure. "
                    f"Expected path: {self._model_path}/{self._cache_model_dir}/snapshots/. "
                    f"This may indicate a Hugging Face cache structure change. "
                    f"Please report this issue."
                )

    def load(self) -> None:
        self._ensure_models()

    def unload(self) -> None:
        with self._rwlock.gen_wlock():
            if self._models is not None:
                self._models = None

    def generate(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> str:
        models = self._ensure_models()
        if device_number is None:
            model_index = self._get_device_number_to_index()[0]
        else:
            model_index = self._get_device_number_to_index()[device_number]

        if model_index is None:
            raise ValueError(f"Device number {device_number} is not enabled.")

        tokenizer = models.tokenizer
        llm_model = models.llms[model_index]
        config = _SIZE_CONFIGS[size]

        with self._rwlock.gen_rlock():
            with InferWithInterruption(llm_model, context) as infer:
                text_result = infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=str(output_path),
                    base_size=config.base_size,
                    image_size=config.image_size,
                    crop_mode=config.crop_mode,
                    save_results=True,
                    test_compress=True,
                    eval_mode=True,
                )
            return text_result

    def _ensure_models(self) -> _Models:
        check_env()
        import torch

        with self._rwlock.gen_rlock():
            if self._models is not None:
                return self._models

        with self._rwlock.gen_wlock():
            if self._models is not None:
                return self._models

            device_number_to_index = self._get_device_number_to_index()
            if len(device_number_to_index) == 0:
                raise RuntimeError("No CUDA devices available")

            name_or_path = self._model_name
            cache_dir: str | None = None

            if self._local_only:
                name_or_path = self._find_pretrained_path()
                if name_or_path is None:
                    raise ValueError(
                        format_local_model_not_found_message(
                            self._model_path,
                            self._cache_model_dir,
                            self._model_name,
                        )
                    )
            else:
                cache_dir = self._cache_dir()

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=name_or_path,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=self._local_only,
            )
            llm_models: list[AutoModel] = []
            for device_number, model_index in enumerate(device_number_to_index):
                if model_index is None:
                    continue
                model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path=name_or_path,
                    _attn_implementation=_ATTN_IMPLEMENTATION,
                    trust_remote_code=True,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                    local_files_only=self._local_only,
                )
                model = model.to(torch.bfloat16).cuda(device_number)
                llm_models.append(preprocess_model(model))

            self._models = _Models(
                tokenizer=tokenizer,
                llms=llm_models,
            )
            return self._models

    def _cache_dir(self) -> str | None:
        if self._model_path is not None:
            return str(self._model_path)
        return None

    def _find_pretrained_path(self) -> str | None:
        assert self._model_path is not None
        return resolve_local_pretrained_path(
            self._model_path,
            self._cache_model_dir,
        )

    def _get_device_number_to_index(self) -> list[int | None]:
        if self._device_number_to_index is None:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if self._enable_devices_numbers is None:
                    self._device_number_to_index = list(range(device_count))
                else:
                    next_model_index: int = 0
                    device_number_to_index: list[int | None] = [None] * device_count
                    for enable_device_number in sorted(
                        list(set(self._enable_devices_numbers))
                    ):
                        if (
                            enable_device_number < 0
                            or enable_device_number >= device_count
                        ):
                            raise ValueError(
                                f"Invalid device number {enable_device_number}, "
                                f"your system has {device_count} CUDA devices."
                            )
                        device_number_to_index[enable_device_number] = next_model_index
                        next_model_index += 1

                    if next_model_index == 0:
                        raise ValueError("No devices are enabled for model loading.")
                    self._device_number_to_index = device_number_to_index
            else:
                self._device_number_to_index = []

            self._enable_devices_numbers = None

        return self._device_number_to_index
