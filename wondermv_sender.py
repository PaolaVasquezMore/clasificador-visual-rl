"""
Script para WonderMV — usa USB serial directo (sin UART externo)
Cargar con CanMV IDE: Herramientas → Guardar como main.py
"""

import sensor
import image
import lcd
import time
import sys

# ── Cámara ──────────────────────────────────────────────────
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

# ── LCD ─────────────────────────────────────────────────────
lcd.init()
lcd.clear(lcd.WHITE)

# ── Rangos LAB calibrados ────────────────────────────────────
COLORES = {
    "red":    (25, 75,  30, 127,  -5, 35),
    "green":  (12, 43, -60,  -3, -10, 40),
    "yellow": (44, 85, -27,   5,   0, 50),
}

UMBRAL_AREA = 3000

def detectar_color(img):
    mejor      = None
    mejor_area = 0
    for nombre, rango in COLORES.items():
        blobs = img.find_blobs(
            [rango],
            pixels_threshold=UMBRAL_AREA,
            area_threshold=UMBRAL_AREA,
            merge=True
        )
        if blobs:
            area = max(b.pixels() for b in blobs)
            if area > mejor_area:
                mejor_area = area
                mejor      = nombre
    return mejor

# ── Bucle principal ──────────────────────────────────────────
ultimo_envio = 0
INTERVALO_MS = 300

while True:
    img   = sensor.snapshot()
    color = detectar_color(img)

    now = time.ticks_ms()
    if time.ticks_diff(now, ultimo_envio) >= INTERVALO_MS:
        if color:
            msg = "COLOR:" + color + "\n"
        else:
            msg = "COLOR:none\n"
        # Enviar por USB serial al PC
        sys.stdout.write(msg)
        ultimo_envio = now

    # Mostrar en LCD
    if color:
        label = "Color: " + color
    else:
        label = "Color: none"
    img.draw_string(5, 5, label, scale=2)
    lcd.display(img)
