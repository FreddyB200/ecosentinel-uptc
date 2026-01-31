# CODEBOOK - Dataset Eficiencia Energética UPTC Boyacá v2

## Descripción General

Dataset de consumo energético y de agua de la UPTC con desglose por **sectores**: Comedores, Salones, Laboratorios, Auditorios y Oficinas.

**Periodo:** Enero 2018 - Octubre 2025  
**Granularidad:** Horaria  
**Registros:** ~275,000  
**Columnas:** 26

---

## Archivos

### 1. consumos_uptc.csv (TABLA PRINCIPAL)

#### Variables de Identificación
| Columna | Tipo | Descripción |
|---------|------|-------------|
| reading_id | int | ID único |
| timestamp | datetime | Fecha y hora |
| sede | string | Tunja, Duitama, Sogamoso, Chiquinquirá |
| sede_id | string | ID sede |

#### Variables de Energía Total
| Columna | Tipo | Descripción | Notas |
|---------|------|-------------|-------|
| energia_total_kwh | float | Suma de todos los sectores | ⚠️ Outliers |
| potencia_total_kw | float | Potencia pico total | |
| co2_kg | float | Emisiones CO2 | ⚠️  missing |

#### Variables de Energía por Sector
| Columna | Tipo | Descripción | Notas |
|---------|------|-------------|-------|
| energia_comedor_kwh | float | Consumo comedores/cafeterías | ⚠️ missing |
| energia_salones_kwh | float | Consumo salones de clase | |
| energia_laboratorios_kwh | float | Consumo laboratorios | ⚠️ Outliers |
| energia_auditorios_kwh | float | Consumo auditorios | ⚠️  missing |
| energia_oficinas_kwh | float | Consumo oficinas admin | |

#### Variables de Agua y Contexto
| Columna | Tipo | Descripción | Notas |
|---------|------|-------------|-------|
| agua_litros | float | Consumo de agua | ⚠️  missing |
| temperatura_exterior_c | float | Temperatura exterior | ⚠️ missing |
| ocupacion_pct | float | Ocupación estimada (%) | ⚠️  missing |

#### Variables Temporales
| Columna | Tipo | Descripción |
|---------|------|-------------|
| hora | int | 0-23 |
| dia_semana | int | 0=Lunes, 6=Domingo |
| dia_nombre | string | Nombre del día |
| mes | int | 1-12 |
| trimestre | int | 1-4 |
| año | int | 2018-2025 |
| periodo_academico | string | ⚠️ inconsistencias |
| es_fin_semana | bool | Sábado o domingo |
| es_festivo | bool | Festivo colombiano |
| es_semana_parciales | bool | Semana de parciales |
| es_semana_finales | bool | Semana de finales |

---


## Distribución por Sede

| Sede | Comedores | Salones | Labs | Auditorios | Oficinas |
|------|-----------|---------|------|------------|----------|
| Tunja | 12% | 25% | 30% | 8% | 25% |
| Duitama | 10% | 28% | 32% | 7% | 23% |
| Sogamoso | 10% | 26% | **35%** | 6% | 23% |
| Chiquinquirá | 8% | **35%** | 20% | 10% | 27% |

*Sogamoso tiene más laboratorios (Ing. Minas, Metalurgia)*
*Chiquinquirá tiene más salones (sede más pequeña, menos labs)*


## Preguntas de Análisis Sugeridas

### Por Sector
1. ¿Qué sector consume más energía en cada sede?
2. ¿Los laboratorios mantienen consumo base consistente 24/7?
3. ¿El pico de comedores coincide exactamente con 12-2pm?
4. ¿Se puede predecir cuándo habrá evento en auditorios?

### Eficiencia
5. ¿Qué sectores tienen oportunidad de reducción en horarios no-pico?
6. ¿Los salones se apagan correctamente en la noche?
7. ¿El consumo de oficinas baja apropiadamente los fines de semana?

### Predicción
8. ¿Qué features son más predictivos para cada sector?
9. ¿Es posible modelar el consumo de auditorios (eventos aleatorios)?
10. ¿Cómo afectó la pandemia a cada sector de forma diferente?

---

*Dataset sintético basado en datos históricos reales de la UPTC 2018-2021.*
*Problemas de calidad son intencionales para propósitos educativos.*
