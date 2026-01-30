# Arquitectura EcoSentinel UPTC

## Diagrama de componentes

```
┌──────────────────────────────────────────────────────────┐
│                    CAPA DE USUARIO                        │
│                                                          │
│  ┌──────────────┐         ┌──────────────────────────┐   │
│  │  WhatsApp     │         │  Dashboard Streamlit      │   │
│  │  (Principal)  │         │  (Secundario - Técnicos)  │   │
│  └──────┬───────┘         └───────────┬──────────────┘   │
└─────────┼─────────────────────────────┼──────────────────┘
          │                             │
┌─────────┼─────────────────────────────┼──────────────────┐
│         ▼          AUTOMATIZACIÓN     │                   │
│  ┌──────────────┐                     │                   │
│  │ Evolution API │                    │                   │
│  └──────┬───────┘                     │                   │
│         ▼                             │                   │
│  ┌──────────────┐                     │                   │
│  │  n8n Workflow │                    │                   │
│  │  ├─ Whisper   │                    │                   │
│  │  └─ GPT-4o   │                    │                   │
│  └──────┬───────┘                     │                   │
└─────────┼─────────────────────────────┼──────────────────┘
          │                             │
┌─────────▼─────────────────────────────▼──────────────────┐
│                    BACKEND (FastAPI)                       │
│                                                          │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐             │
│  │  /api/chat  │  │ /predict │  │ /anomaly │  ...        │
│  └─────┬──────┘  └────┬─────┘  └────┬─────┘             │
│        │              │              │                    │
│  ┌─────▼──────────────▼──────────────▼─────┐             │
│  │           Lógica de negocio              │             │
│  │  ┌──────────┐ ┌────────┐ ┌───────────┐  │             │
│  │  │ Prophet  │ │ Z-Score│ │  Gemini   │  │             │
│  │  │ (20 mod.)│ │ Detect.│ │  LLM      │  │             │
│  │  └──────────┘ └────────┘ └───────────┘  │             │
│  │  ┌──────────────────────────────────┐   │             │
│  │  │  Matplotlib (Chart Generation)   │   │             │
│  │  └──────────────────────────────────┘   │             │
│  └─────────────────────┬───────────────────┘             │
└────────────────────────┼─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                    DATOS                                  │
│  ┌─────────────────────────────────────────┐             │
│  │         Supabase (PostgreSQL)            │             │
│  │  ├─ consumos (270k registros)           │             │
│  │  ├─ predicciones                        │             │
│  │  ├─ anomalias                           │             │
│  │  └─ recomendaciones                     │             │
│  └─────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
```

## Flujo de datos

### WhatsApp → Respuesta
1. Usuario envía audio/texto por WhatsApp
2. Evolution API recibe y lo enruta a n8n
3. n8n transcribe audio (Whisper) si aplica
4. GPT-4o-mini detecta intención (consumo, predicción, anomalía, etc.)
5. n8n envía HTTP POST a FastAPI `/api/chat`
6. FastAPI orquesta la respuesta según la intención
7. Respuesta JSON con texto + URL gráfico + recomendaciones
8. n8n formatea y envía respuesta por WhatsApp

### Modelo Prophet
- 20 modelos independientes (4 sedes x 5 sectores)
- Entrenados con datos 2018-2025
- Regresores: temperatura, ocupación, fin de semana
- Predicción multi-horizonte: 1 hora a 7 días

### Detección de anomalías
- Z-Score para anomalías estadísticas (umbral configurable)
- Patrones: consumo nocturno elevado, fines de semana sin reducción
- Cálculo de ahorro potencial en kWh, COP y CO2

## Despliegue

```
Hostinger VPS (8 CPU, 16GB RAM, 100GB SSD)
├── Docker Compose
│   ├── backend (FastAPI) :8000
│   └── frontend (Streamlit) :8501
├── n8n (Docker separado) :5678
└── Evolution API :8080
```
