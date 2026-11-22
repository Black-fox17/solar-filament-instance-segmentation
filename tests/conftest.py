"""Pytest configuration.

Importing torch before scientific/plugin modules avoids Windows DLL load-order
conflicts (WinError 1114 on c10.dll) when third-party pytest plugins are installed.
"""

try:
    import torch  # noqa: F401
except Exception:
    pass
