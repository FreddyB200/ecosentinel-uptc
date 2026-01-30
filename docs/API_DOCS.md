# EcoSentinel UPTC - Documentación API

## Base URL
```
http://localhost:8000
```

## Autenticación
La API actualmente no requiere autenticación. En producción se recomienda agregar API keys.

---

## Endpoints

### Health Check
```
GET /health
```
Respuesta:
```json
{"status": "healthy", "service": "ecosentinel-api", "version": "1.0.0"}
```

---

### Chat (WhatsApp)
```
POST /api/chat
```

Endpoint principal para interacción desde n8n/WhatsApp.

**Body:**
| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| message | string | Si | Mensaje del usuario |
| user_id | string | No | ID del usuario (default: "anonymous") |
| intent | string | No | Intención detectada por n8n |
| sede | string | No | Sede UPTC (default: Tunja) |
| sector | string | No | Sector (default: Comedores) |

**Intenciones soportadas:**
- `consumo_historico` - Consultar consumo pasado
- `prediccion` - Generar predicción
- `anomalias` - Detectar anomalías
- `recomendaciones` - Obtener recomendaciones
- `comparacion` - Comparar sectores

---

### Predicción
```
POST /api/predict
```

**Body:**
| Campo | Tipo | Requerido | Default | Descripción |
|-------|------|-----------|---------|-------------|
| sede | string | Si | - | Sede UPTC |
| sector | string | Si | - | Sector |
| hours_ahead | int | No | 24 | Horas a predecir (1-168) |
| temperatura | float | No | 18.0 | Temperatura estimada (°C) |
| ocupacion | float | No | 60.0 | Ocupación estimada (%) |
| include_chart | bool | No | true | Incluir URL del gráfico |

---

### Consumo Histórico
```
GET /api/consumption?sede=Tunja&sector=Comedores&dias=7
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| sede | string | Requerido | Sede UPTC |
| sector | string | Requerido | Sector |
| dias | int | 7 | Días hacia atrás (1-365) |
| include_chart | bool | true | Incluir gráfico |

---

### Comparación entre Sectores
```
GET /api/consumption/compare?sede=Tunja&dias=7
```

---

### Anomalías
```
GET /api/anomalies?sede=Tunja&sector=Comedores
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| sede | string | Requerido | Sede UPTC |
| sector | string | Requerido | Sector |
| dias | int | 30 | Días hacia atrás |
| threshold | float | 2.5 | Umbral Z-Score (1.5-4.0) |
| include_chart | bool | true | Incluir gráfico |

---

### Recomendaciones
```
POST /api/recommendations
```

**Body:**
| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| sede | string | Si | Sede UPTC |
| sector | string | Si | Sector |

---

## Valores válidos

**Sedes:** `Tunja`, `Duitama`, `Sogamoso`, `Chiquinquirá`

**Sectores:** `Comedores`, `Salones`, `Laboratorios`, `Auditorios`, `Oficinas`

## Códigos de error

| Código | Descripción |
|--------|-------------|
| 400 | Parámetros inválidos (sede/sector no reconocido) |
| 404 | Modelo o datos no encontrados |
| 500 | Error interno del servidor |
