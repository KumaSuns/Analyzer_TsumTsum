from pathlib import Path


class TsumRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def list_tsum_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
