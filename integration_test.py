"""Basic integration smoke test for Docker handoff environment."""

from pathlib import Path

import torch


def main() -> None:
    print("Running integration smoke test")
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")

    output_root = Path("/outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "integration_test.ok"
    marker.write_text("smoke_ok\n", encoding="utf-8")
    print(f"Wrote marker: {marker}")


if __name__ == "__main__":
    main()
