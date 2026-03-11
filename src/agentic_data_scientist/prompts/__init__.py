"""Prompt templates and loading utilities."""

from pathlib import Path
from typing import Optional


def load_prompt(name: str, domain: Optional[str] = None) -> str:
    """
    Load prompt template by name.

    Parameters
    ----------
    name : str
        Prompt name (e.g., 'plan_generator', 'coding_review')
    domain : str, optional
        Optional domain namespace (e.g., 'bioinformatics')

    Returns
    -------
    str
        Prompt template string

    Raises
    ------
    FileNotFoundError
        If the prompt file doesn't exist
    """
    prompts_dir = Path(__file__).parent

    base_path = prompts_dir / "base" / f"{name}.md"

    if domain:
        domain_path = prompts_dir / "domain" / domain / f"{name}.md"
        if domain_path.exists():
            return domain_path.read_text()
        # Fall back to base if domain-specific prompt doesn't exist

    if not base_path.exists():
        raise FileNotFoundError(f"Prompt not found: {base_path}")

    return base_path.read_text()


__all__ = ["load_prompt"]
