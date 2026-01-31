"""
Router principal para el endpoint /chat.
Recibe mensajes desde n8n (WhatsApp) y orquesta la respuesta con predicciones,
gráficos y recomendaciones.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.config import SEDES, SECTORES
from api.utils.db import get_consumos, get_consumos_resumen
from api.utils.charts import generate_consumption_chart, generate_prediction_chart
from api.ml.predictor import model_manager
from api.ml.anomaly_detector import generate_anomaly_report
from api.llm.recommender import generate_recommendations

logger = logging.getLogger("ecosentinel.chat")
router = APIRouter()


class ChatRequest(BaseModel):
    """Esquema de solicitud del chat (viene de n8n)."""
    message: str
    user_id: str = "anonymous"
    intent: Optional[str] = None
    sede: Optional[str] = None
    sector: Optional[str] = None


class RecommendationItem(BaseModel):
    """Esquema de una recomendación individual."""
    titulo: str
    descripcion: str
    ahorro_estimado_kwh: float = 0
    ahorro_estimado_cop: float = 0
    reduccion_co2_kg: float = 0
    prioridad: str = "media"
    plazo: str = "corto"


class ChatResponse(BaseModel):
    """Esquema de respuesta del chat."""
    text: str
    chart_url: Optional[str] = None
    recommendations: list[RecommendationItem] = []
    intent_detected: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal para interacción via WhatsApp.

    Flujo:
    1. n8n detecta intención del usuario (GPT-4o-mini)
    2. Envía request con intent, sede y sector
    3. Este endpoint orquesta la respuesta según la intención
    """
    logger.info(f"Chat request: user={request.user_id}, intent={request.intent}, "
                f"sede={request.sede}, sector={request.sector}")

    # Validar sede y sector
    sede = _normalize_sede(request.sede)
    sector = _normalize_sector(request.sector)

    intent = (request.intent or "").lower().strip()

    try:
        if intent == "consumo_historico":
            return await _handle_consumo(sede, sector)
        elif intent == "prediccion":
            return await _handle_prediccion(sede, sector)
        elif intent == "anomalias":
            return await _handle_anomalias(sede, sector)
        elif intent == "recomendaciones":
            return await _handle_recomendaciones(sede, sector)
        elif intent == "comparacion":
            return await _handle_comparacion(sede)
        else:
            return await _handle_general(request.message, sede, sector)
    except Exception as e:
        logger.error(f"Error procesando chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error procesando solicitud: {str(e)}")


async def _handle_consumo(sede: str, sector: str) -> ChatResponse:
    """Maneja consultas de consumo histórico."""
    resumen = get_consumos_resumen(sede, sector, dias=7)

    if resumen["registros"] == 0:
        return ChatResponse(
            text=f"No se encontraron datos de consumo para {sector} en {sede}.",
            intent_detected="consumo_historico",
        )

    # Generar gráfico
    df = get_consumos(sede, sector)
    chart_url = None
    if not df.empty:
        chart_url = generate_consumption_chart(df, sede, sector)

    text = (
        f"Consumo energético - {sede} / {sector} (últimos 7 días)\n\n"
        f"Promedio: {resumen['promedio']:,.1f} kWh\n"
        f"Pico máximo: {resumen['maximo']:,.1f} kWh\n"
        f"Mínimo: {resumen['minimo']:,.1f} kWh\n"
        f"Total período: {resumen['total']:,.1f} kWh\n"
        f"Registros: {resumen['registros']:,}"
    )

    return ChatResponse(text=text, chart_url=chart_url, intent_detected="consumo_historico")


async def _handle_prediccion(sede: str, sector: str) -> ChatResponse:
    """Maneja solicitudes de predicción."""
    predictions = model_manager.predict(sede, sector, hours_ahead=24)

    if predictions is None:
        return ChatResponse(
            text=f"Modelo de predicción para {sector} en {sede} no disponible.",
            intent_detected="prediccion",
        )

    # Obtener histórico reciente para el gráfico
    df_hist = get_consumos(sede, sector)
    chart_url = None
    if not df_hist.empty:
        chart_url = generate_prediction_chart(df_hist, predictions, sede, sector)

    promedio_pred = predictions["yhat"].mean()
    maximo_pred = predictions["yhat"].max()
    minimo_pred = predictions["yhat"].min()

    text = (
        f"Predicción 24h - {sede} / {sector}\n\n"
        f"Consumo promedio esperado: {promedio_pred:,.1f} kWh\n"
        f"Pico esperado: {maximo_pred:,.1f} kWh\n"
        f"Mínimo esperado: {minimo_pred:,.1f} kWh\n"
        f"Total estimado 24h: {predictions['yhat'].sum():,.1f} kWh"
    )

    return ChatResponse(text=text, chart_url=chart_url, intent_detected="prediccion")


async def _handle_anomalias(sede: str, sector: str) -> ChatResponse:
    """Maneja consultas sobre anomalías."""
    df = get_consumos(sede, sector, limit=2000)
    if df.empty:
        return ChatResponse(
            text=f"No hay datos suficientes para detectar anomalías en {sector} de {sede}.",
            intent_detected="anomalias",
        )

    reporte = generate_anomaly_report(df, sede, sector)

    text = f"Reporte de anomalías - {sede} / {sector}\n\n"
    text += f"Anomalías detectadas: {reporte['total_anomalias']}\n"

    if reporte["patrones"]:
        text += "\nPatrones detectados:\n"
        for p in reporte["patrones"]:
            text += f"- {p['descripcion']}\n"

    if reporte["ahorro_potencial_kwh"] > 0:
        text += (
            f"\nAhorro potencial: {reporte['ahorro_potencial_kwh']:,.0f} kWh "
            f"({reporte['ahorro_potencial_co2_kg']:,.1f} kg CO2)"
        )

    return ChatResponse(text=text, intent_detected="anomalias")


async def _handle_recomendaciones(sede: str, sector: str) -> ChatResponse:
    """Maneja solicitudes de recomendaciones."""
    resumen = get_consumos_resumen(sede, sector)
    df = get_consumos(sede, sector)

    # Obtener anomalías y patrones para contexto
    anomalias_data = []
    patrones_data = []
    if not df.empty:
        reporte = generate_anomaly_report(df, sede, sector)
        anomalias_data = reporte.get("anomalias", [])
        patrones_data = reporte.get("patrones", [])

    recs = await generate_recommendations(
        sede=sede,
        sector=sector,
        consumo_promedio=resumen.get("promedio", 0),
        consumo_maximo=resumen.get("maximo", 0),
        anomalias=anomalias_data,
        patrones=patrones_data,
    )

    text = f"Recomendaciones - {sede} / {sector}\n"
    rec_items = []
    for i, r in enumerate(recs, 1):
        text += (
            f"\n{i}. {r['titulo']}\n"
            f"   {r['descripcion']}\n"
            f"   Ahorro estimado: {r.get('ahorro_estimado_kwh', 0):,.0f} kWh "
            f"(${r.get('ahorro_estimado_cop', 0):,.0f} COP)\n"
            f"   Prioridad: {r.get('prioridad', 'media')}\n"
        )
        rec_items.append(RecommendationItem(**r))

    return ChatResponse(text=text, recommendations=rec_items, intent_detected="recomendaciones")


async def _handle_comparacion(sede: str) -> ChatResponse:
    """Maneja comparaciones entre sectores de una sede."""
    datos = {}
    for sector in SECTORES:
        resumen = get_consumos_resumen(sede, sector, dias=7)
        datos[sector] = resumen.get("promedio", 0)

    from api.utils.charts import generate_comparison_chart
    chart_url = generate_comparison_chart(datos, titulo=f"Consumo por sector - {sede}")

    text = f"Comparación de consumo por sector - {sede} (últimos 7 días)\n\n"
    for sector, promedio in sorted(datos.items(), key=lambda x: x[1], reverse=True):
        text += f"- {sector}: {promedio:,.1f} kWh promedio\n"

    return ChatResponse(text=text, chart_url=chart_url, intent_detected="comparacion")


async def _handle_general(message: str, sede: str, sector: str) -> ChatResponse:
    """Maneja mensajes sin intención clara."""
    text = (
        "Puedo ayudarte con la gestión energética de la UPTC. Intenta preguntar sobre:\n\n"
        "- Consumo histórico de una sede o sector\n"
        "- Predicción de consumo para las próximas horas\n"
        "- Anomalías detectadas en el consumo\n"
        "- Recomendaciones para reducir el consumo\n"
        "- Comparación entre sectores\n\n"
        "Ejemplo: \"Muéstrame el consumo de comedores en Tunja\""
    )
    return ChatResponse(text=text, intent_detected="general")


def _normalize_sede(sede: Optional[str]) -> str:
    """Normaliza el nombre de sede. Default: Tunja."""
    if not sede:
        return "Tunja"
    for s in SEDES:
        if s.lower() == sede.lower().strip():
            return s
    return "Tunja"


def _normalize_sector(sector: Optional[str]) -> str:
    """Normaliza el nombre de sector. Default: Comedores."""
    if not sector:
        return "Comedores"
    for s in SECTORES:
        if s.lower() == sector.lower().strip():
            return s
    return "Comedores"
