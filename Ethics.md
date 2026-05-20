# Ética, Sesgos y Transparencia del Modelo

## Sesgos identificados en el dataset

**Género:** El dataset incluye la variable `gender` como predictor. Esto puede llevar a que el modelo tome decisiones diferentes basadas en el género del cliente sin una justificación técnica válida, lo cual representa un riesgo de discriminación.

**Clientes mayores:** La variable `SeniorCitizen` puede hacer que el modelo sea menos preciso para adultos mayores si este grupo está subrepresentado en los datos de entrenamiento.

**Sesgo histórico:** Los datos reflejan decisiones comerciales pasadas. Si en el pasado ciertos grupos de clientes recibieron peor servicio, el modelo aprenderá esos patrones y los reproducirá.

---

## Variables sensibles

| Variable | Riesgo |
|----------|--------|
| gender | Discriminación por género |
| SeniorCitizen | Discriminación por edad |
| PaymentMethod | Puede correlacionar con nivel socioeconómico |

---

## Limitaciones del modelo

- El modelo fue entrenado con datos de una sola empresa, por lo que puede no generalizar bien a otras empresas de telecomunicaciones.
- Un Recall de 0.51 significa que el modelo falla en detectar aproximadamente la mitad de los clientes que realmente abandonan el servicio.
- No se aplicaron técnicas de balanceo de clases, lo que puede afectar el rendimiento en la clase minoritaria (Churn = Yes).

---

## Recomendaciones

- No usar `gender` ni `SeniorCitizen` como predictores principales en un entorno de producción real.
- Monitorear el rendimiento del modelo por grupos demográficos