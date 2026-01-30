"""
Integración con Gemini para generar recomendaciones energéticas accionables.
"""

import logging
import json
from typing import Optional

import google.generativeai as genai

from api.config import GEMINI_API_KEY, PRECIO_KWH_COP, CO2_FACTOR

logger = logging.getLogger("ecosentinel.recommender")

_model = None


def _get_model():
    """Inicializa el modelo Gemini de forma lazy."""
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY no configurada")
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Modelo Gemini inicializado")
    return _model


async def generate_recommendations(
    sede: str,
    sector: str,
    consumo_promedio: float,
    consumo_maximo: float,
    anomalias: list[dict] = None,
    patrones: list[dict] = None,
) -> list[dict]:
    """
    Genera recomendaciones personalizadas usando Gemini.

    Args:
        sede: Nombre de la sede UPTC
        sector: Nombre del sector
        consumo_promedio: Consumo promedio reciente (kWh)
        consumo_maximo: Consumo máximo reciente (kWh)
        anomalias: Lista de anomalías detectadas
        patrones: Lista de patrones ineficientes detectados

    Returns:
        Lista de recomendaciones con título, descripción, ahorro estimado y prioridad
    """
    anomalias_text = ""
    if anomalias:
        anomalias_text = f"\nAnomalías detectadas: {len(anomalias)}"
        for a in anomalias[:5]:
            anomalias_text += f"\n- {a.get('tipo', 'N/A')}: {a.get('valor_actual', 0)} kWh (esperado: {a.get('valor_esperado', 0)} kWh)"

    patrones_text = ""
    if patrones:
        for p in patrones:
            patrones_text += f"\n- {p.get('tipo', 'N/A')}: {p.get('descripcion', '')}"

    prompt = f"""Eres un experto en eficiencia energética para universidades en Colombia.

CONTEXTO:
- Universidad: UPTC (Universidad Pedagógica y Tecnológica de Colombia)
- Sede: {sede}
- Sector: {sector}
- Consumo promedio reciente: {consumo_promedio:.1f} kWh/hora
- Consumo máximo registrado: {consumo_maximo:.1f} kWh/hora
- Precio kWh en Colombia: ${PRECIO_KWH_COP:,.0f} COP
- Factor emisión CO2: {CO2_FACTOR} kg CO2/kWh
{anomalias_text}
{patrones_text}

TAREA: Genera exactamente 3 recomendaciones específicas y accionables para reducir el consumo energético.

Responde SOLO con un JSON array con esta estructura exacta:
[
  {{
    "titulo": "Título corto de la recomendación",
    "descripcion": "Descripción detallada de la acción a tomar, incluyendo pasos específicos",
    "ahorro_estimado_kwh": 100,
    "ahorro_estimado_cop": 65000,
    "reduccion_co2_kg": 12.6,
    "prioridad": "alta",
    "plazo": "corto"
  }}
]

Notas:
- prioridad: "alta", "media", o "baja"
- plazo: "inmediato", "corto" (1-3 meses), "mediano" (3-12 meses)
- Los valores de ahorro deben ser realistas y calculados con base en los datos
- Las recomendaciones deben ser específicas para el tipo de sector ({sector})
"""

    try:
        model = _get_model()
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Limpiar markdown si Gemini lo envuelve
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        recommendations = json.loads(text)
        logger.info(f"Generadas {len(recommendations)} recomendaciones para {sede}/{sector}")
        return recommendations

    except json.JSONDecodeError as e:
        logger.error(f"Error parseando respuesta Gemini: {e}")
        return _fallback_recommendations(sede, sector, consumo_promedio)
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        return _fallback_recommendations(sede, sector, consumo_promedio)


def _fallback_recommendations(
    sede: str,
    sector: str,
    consumo_promedio: float,
) -> list[dict]:
    """Recomendaciones estáticas de respaldo si Gemini falla."""
    ahorro_10pct = consumo_promedio * 0.10 * 24 * 30  # mensual
    return [
        {
            "titulo": f"Optimizar horarios de uso en {sector}",
            "descripcion": (
                f"Reducir el consumo energético en {sector} de la sede {sede} "
                "ajustando los horarios de encendido/apagado de equipos según "
                "la ocupación real del espacio."
            ),
            "ahorro_estimado_kwh": round(ahorro_10pct, 0),
            "ahorro_estimado_cop": round(ahorro_10pct * PRECIO_KWH_COP, 0),
            "reduccion_co2_kg": round(ahorro_10pct * CO2_FACTOR, 1),
            "prioridad": "alta",
            "plazo": "inmediato",
        },
        {
            "titulo": "Implementar sensores de presencia",
            "descripcion": (
                "Instalar sensores de movimiento para controlar iluminación "
                "y climatización automáticamente según ocupación."
            ),
            "ahorro_estimado_kwh": round(ahorro_10pct * 1.5, 0),
            "ahorro_estimado_cop": round(ahorro_10pct * 1.5 * PRECIO_KWH_COP, 0),
            "reduccion_co2_kg": round(ahorro_10pct * 1.5 * CO2_FACTOR, 1),
            "prioridad": "media",
            "plazo": "corto",
        },
        {
            "titulo": "Auditoría de equipos de alto consumo",
            "descripcion": (
                "Identificar y reemplazar equipos obsoletos o ineficientes "
                "por alternativas de bajo consumo energético."
            ),
            "ahorro_estimado_kwh": round(ahorro_10pct * 2, 0),
            "ahorro_estimado_cop": round(ahorro_10pct * 2 * PRECIO_KWH_COP, 0),
            "reduccion_co2_kg": round(ahorro_10pct * 2 * CO2_FACTOR, 1),
            "prioridad": "media",
            "plazo": "mediano",
        },
    ]
