from pathlib import Path


def ensure_model_available(model_name, project_root=None):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name must be a non-empty Hugging Face repository ID.")

    model_parts = model_name.split("/")
    if any(
        part in {"", ".", ".."} or "\\" in part
        for part in model_parts
    ):
        raise ValueError(f"Invalid Hugging Face repository ID: {model_name!r}")

    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    model_directory = root / "data" / "models" / Path(*model_parts)
    completion_marker = model_directory / ".chatveritas-download-complete"

    if completion_marker.is_file():
        return str(model_directory)

    from huggingface_hub import snapshot_download

    model_directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(model_directory),
    )
    completion_marker.touch()
    return str(model_directory)