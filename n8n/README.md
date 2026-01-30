# n8n Workflows - EcoSentinel UPTC

## Workflow principal: ecosentinel_workflow.json

### Flujo
1. **Trigger:** Webhook de Evolution API (mensaje WhatsApp entrante)
2. **Audio check:** Si es audio → Whisper API para transcripción
3. **Intent Detection:** GPT-4o-mini clasifica intención del usuario
4. **API Call:** HTTP Request a FastAPI `/api/chat`
5. **Response Format:** Formatear texto + imagen para WhatsApp
6. **Send:** Evolution API envía respuesta al usuario

### Intenciones detectadas
- `consumo_historico` - "Muéstrame el consumo de..."
- `prediccion` - "Cuánto se va a consumir..."
- `anomalias` - "Hay algo raro en..."
- `recomendaciones` - "Cómo puedo reducir..."
- `comparacion` - "Compara las sedes..."

### Configuración requerida
- Evolution API URL y API Key
- OpenAI API Key (Whisper + GPT-4o-mini)
- FastAPI Backend URL

### Importar workflow
1. Abrir n8n
2. Ir a Workflows → Import
3. Seleccionar `ecosentinel_workflow.json`
4. Configurar credenciales
