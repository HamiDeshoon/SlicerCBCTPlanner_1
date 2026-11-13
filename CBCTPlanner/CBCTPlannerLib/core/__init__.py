"""Core dental CBCT algorithms with lazy optional dependency loading.

All optional third-party libraries (scipy, scikit-image, nibabel, pydicom, SimpleITK)
are imported lazily only when a function requiring them is invoked. Importing this
package requires only Python's standard library and numpy.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

logger = logging.getLogger("CBCTPlanner.core")


def ensure_optional_package(
    import_name: str,
    pip_name: Optional[str] = None,
    feature_name: str = "this feature",
    prompt_install: bool = True,
) -> Any:
    """Import an optional package lazily, prompting via slicer.util.pip_install if inside Slicer."""
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        pkg = pip_name or import_name
        try:
            import slicer  # type: ignore

            if prompt_install and getattr(slicer, "app", None) is not None:
                msg = (
                    f"Optional Python package '{pkg}' is required for {feature_name}.\n\n"
                    f"Would you like 3D Slicer to install '{pkg}' now?"
                )
                if slicer.util.confirmOkCancelDisplay(msg, windowTitle="Install Optional Dependency"):
                    slicer.util.pip_install(pkg)
                    return importlib.import_module(import_name)
        except Exception as install_exc:
            logger.warning("Optional package install skipped or failed for %s: %s", pkg, install_exc)
        raise ImportError(
            f"Optional dependency '{pkg}' is required for {feature_name} but is not installed."
        ) from exc
