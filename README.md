# Clasificador Visual con Aprendizaje por Refuerzo

Proyecto Final — Inteligencia Artificial  
Universidad Técnica Federico Santa María 2026-1

## Integrantes
- Patricio Guerra Ortiz — Integración hardware y salida visual
- Paola Vásquez Moreno — Implementación algorítmica
- Víctor Vegas Calderón — Documentación, métricas e implementación de código
- Maximiliano Estay — Documentación y métricas
**Profesora:** Heilym Ramírez Rico

## Descripción
Sistema capaz de clasificar objetos físicos según su color de forma autónoma usando el módulo WonderMV como entrada visual y un agente Q-learning tabular como mecanismo de aprendizaje por refuerzo. El sistema evita las reglas estáticas y aprende asociaciones basándose en recompensas estocásticas.

## Formulación del Problema (MDP)
- **Espacio de Estados (S):** Discreto. 4 estados (Rojo, Verde, Amarillo, Inactivo) captados directamente de los blobs visuales.
- **Espacio de Acciones (A):** Discreto. 4 posibles paneles a desplegar en terminal.
- **Recompensa (R):** Asimétrica. +[6.0 a 10.0] por acierto, -[1.0 a 5.0] por fallo.
- **Política de Exploración:** $\epsilon$-greedy con factor de decaimiento.

## Flujo del sistema
WonderMV (procesamiento en vivo) → USB Serial → Entorno RL (Python) → Actualización Q-Table → Despliegue en consola colorama.

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
py main.py
```

### Opciones de uso
1. Entrenar un modelo, permite entrenar un nuevo modelo que será guardado para posterior utilizando pickle
2. Permite utilizar en tiempo real el modelo entrenado y guardado con anterioridad
3. Muestra las metricas del modelo para evaluar su desempeño
4. Salida del programa

## Archivos
- `color_rl_system.py` — Agente RL principal (PC)
- `wondermv_sender.py` — Script MicroPython para WonderMV
- `main.py` — Control de flujo de uso del programa

## Resultados
Se plantea demostrar convergencia estable a partir del episodio 50, logrando estabilizar la media móvil de recompensas en valores positivos constantes y demostrando que los valores máximos de la Q-Table se alinean perfectamente en la diagonal principal (estado correcto = acción correcta).

## Hardware
- Módulo WonderMV con cámara HD 2MP y pantalla táctil
- Chip K210 con procesamiento visual en vivo
- Comunicación serial USB al PC
