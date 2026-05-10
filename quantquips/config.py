from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _env_path("QUANTQUIPS_DATA_DIR", PROJECT_ROOT / "data")
    llm_provider: str = os.getenv("QUANTQUIPS_LLM_PROVIDER", "disabled").strip().lower()
    llm_model: str = os.getenv("QUANTQUIPS_LLM_MODEL", "")
    llm_base_url: str = os.getenv("QUANTQUIPS_LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("QUANTQUIPS_LLM_API_KEY", "")

    @property
    def company_data_dir(self) -> Path:
        return self.data_dir / "companyData"

    @property
    def ticker_list_dir(self) -> Path:
        return self.data_dir / "TickerList"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"


def get_settings() -> Settings:
    return Settings()
