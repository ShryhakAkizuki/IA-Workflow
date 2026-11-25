"""Paquete del orquestador de voz (STT + LLM).

La configuración se centraliza en :mod:`src.config` (objeto ``settings``);
ver ``.env.example`` en la raíz del repo para las variables necesarias.
"""

from src.config import Settings, settings

__all__ = ["Settings", "settings"]
