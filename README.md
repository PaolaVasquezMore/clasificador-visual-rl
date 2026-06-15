# Clasificador Visual con Aprendizaje por Refuerzo

Proyecto Final — Inteligencia Artificial  
Universidad Técnica Federico Santa María 2026-1

## Integrantes
- Patricio Guerra Ortiz — Integración hardware y salida visual
- Paola Vásquez Moreno — Implementación algorítmica
- Víctor Vegas Calderón — Documentación y métricas

**Profesora:** Heilym Ramírez Rico

## Descripción
Sistema capaz de clasificar objetos físicos según su color de forma autónoma
usando el módulo WonderMV (K210) como entrada visual y un agente Q-learning
tabular como mecanismo de aprendizaje por refuerzo.

## Flujo del sistema
WonderMV → COLOR:red/green/yellow → color_leido() → Q-table → terminal colorama

## Colores soportados
| Color | Estado ID | Panel terminal |
|-------|-----------|----------------|
| Rojo | 0 | Fondo rojo |
| Verde | 1 | Fondo verde |
| Amarillo | 2 | Fondo amarillo |
| Sin objeto | 3 | Fondo blanco |

## Instalación
```bash
pip install pyserial colorama numpy matplotlib
```

## Uso
1. Cargar `wondermv_sender.py` en la WonderMV con CanMV IDE
2. Conectar la WonderMV al PC por USB
3. Cerrar CanMV IDE
4. Ejecutar:
```bash
py color_rl_system.py
```

## Archivos
- `color_rl_system.py` — Agente RL principal (PC)
- `wondermv_sender.py` — Script MicroPython para WonderMV
- `curvas_entrenamiento.png` — Evidencia gráfica del entrenamiento

## Resultados
El agente converge en promedio en 13 episodios con 76.9% de tasa de aciertos.
La Q-table aprende correctamente la asociación color detectado → panel desplegado.

## Hardware
- Módulo WonderMV con cámara HD 2MP y pantalla táctil
- Chip K210 con procesamiento visual en vivo
- Comunicación serial USB al PC
