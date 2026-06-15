"""
Clasificador Visual con Aprendizaje por Refuerzo
Universidad Técnica Federico Santa María — 2026-1
Integrantes: P. Guerra, P. Vásquez, V. Vegas

FLUJO:
  WonderMV (K210) → serial USB → color_leido() → Red Neuronal (perceptrón)
                                                → Q-table (tabular RL)
                                                → despliegue terminal (colorama)

COLORES: rojo, verde, amarillo
ACCIONES: mostrar panel rojo / verde / amarillo / inactivo
"""

import serial
import serial.tools.list_ports
import numpy as np
import colorama
from colorama import Fore, Back, Style
import time
import random

colorama.init(autoreset=True)

# ─────────────────────────────────────────────────────────────
# 1. COLORES ESTÁTICOS  (rojo, verde, amarillo)
# ─────────────────────────────────────────────────────────────
# La WonderMV manda strings por serial.
# Mapeamos los posibles strings que puede enviar a nuestros IDs.

COLOR_ID = {
    "rojo":     0,
    "verde":    1,
    "amarillo": 2,
    None:       3,   # inactivo / sin objeto
}

ID_COLOR = {v: k for k, v in COLOR_ID.items()}

# Lo que puede enviar la WonderMV (ajustar si el firmware usa otro idioma)
WONDERMV_MAP = {
    "red":    "rojo",
    "green":  "verde",
    "yellow": "amarillo",
    "none":   None,
    "":       None,
}

# Despliegue colorama en terminal del PC
COLOR_DISPLAY = {
    "rojo":     Back.RED    + Fore.WHITE,
    "verde":    Back.GREEN  + Fore.BLACK,
    "amarillo": Back.YELLOW + Fore.BLACK,
    None:       Back.WHITE  + Fore.BLACK,
}

# ─────────────────────────────────────────────────────────────
# 2. FUNCIÓN color_leido()  ←  recibe dato serial de WonderMV
# ─────────────────────────────────────────────────────────────

def detectar_puerto_wondermv() -> str | None:
    """
    Busca automáticamente el puerto serie de la WonderMV.
    Si no lo encuentra retorna None (modo simulado).
    """
    for p in serial.tools.list_ports.comports():
        desc = p.description.lower()
        if any(k in desc for k in ["ch340", "cp210", "usb serial", "wondermv"]):
            return p.device
    return None


def color_leido(linea_serial: str) -> str | None:
    """
    Recibe una línea de texto enviada por la WonderMV vía serial.
    Retorna: 'rojo', 'verde', 'naranja' o None.

    La WonderMV debe estar programada para enviar líneas como:
        COLOR:red\n
        COLOR:green\n
        COLOR:none\n
    """
    linea = linea_serial.strip().lower()
    if linea.startswith("color:"):
        clave = linea.split(":", 1)[1].strip()
        return WONDERMV_MAP.get(clave, None)
    return None   # línea no reconocida → inactivo


# ─────────────────────────────────────────────────────────────
# 3. RED NEURONAL PRIMITIVA  (perceptrón 1→N clases)
# ─────────────────────────────────────────────────────────────

class Perceptron:
    """
    Red de 1 capa con softmax.
    Entrada : one-hot del color leído (4 estados: rojo/verde/naranja/inactivo)
    Salida  : distribución sobre 4 acciones
    """

    def __init__(self, n: int = 4, lr: float = 0.1):
        self.n  = n
        self.lr = lr
        self.W  = np.random.uniform(-0.5, 0.5, (n, n))
        self.b  = np.zeros(n)

    def _softmax(self, z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def forward(self, estado_id: int) -> int:
        x     = np.eye(self.n)[estado_id]
        probs = self._softmax(self.W @ x + self.b)
        return int(np.argmax(probs))

    def actualizar(self, estado_id: int, accion_correcta_id: int, reward: float):
        x      = np.eye(self.n)[estado_id]
        probs  = self._softmax(self.W @ x + self.b)
        target = np.eye(self.n)[accion_correcta_id]
        delta  = (target - probs) * abs(reward)
        self.W += self.lr * np.outer(delta, x) * np.sign(reward)
        self.b += self.lr * delta * np.sign(reward)


# ─────────────────────────────────────────────────────────────
# 4. Q-TABLE TABULAR  (aprendizaje por refuerzo)
# ─────────────────────────────────────────────────────────────

class QTableRL:
    """
    Q-learning tabular.
    Estado  = color leído (0-3)
    Acción  = panel a mostrar (0-3: rojo/verde/naranja/inactivo)
    Reward  : +[6,10] si acierta, −[1,5] si falla
    Listo   : cuando el promedio de los últimos 10 rewards > 0
              y los últimos 5 son todos positivos
    """

    def __init__(self, n: int = 4, alpha: float = 0.3,
                 gamma: float = 0.9, epsilon: float = 0.3):
        self.Q       = np.zeros((n, n))
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

    def elegir_accion(self, estado: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.Q.shape[1] - 1)
        return int(np.argmax(self.Q[estado]))

    def actualizar(self, s: int, a: int, r: float, s2: int):
        td = r + self.gamma * np.max(self.Q[s2]) - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

    def decaer_epsilon(self, factor: float = 0.995, minimo: float = 0.05):
        self.epsilon = max(minimo, self.epsilon * factor)

    def esta_listo(self, historial: list) -> bool:
        if len(historial) < 10:
            return False
        return (np.mean(historial[-10:]) > 0 and
                all(r > 0 for r in historial[-5:]))


# ─────────────────────────────────────────────────────────────
# 5. DESPLIEGUE EN TERMINAL
# ─────────────────────────────────────────────────────────────

def desplegar_color(color: str | None, correcto: bool, ep: int, reward: float):
    estilo  = COLOR_DISPLAY.get(color, COLOR_DISPLAY[None])
    etiq    = color.upper() if color else "INACTIVO"
    simbolo = "✔" if correcto else "✘"
    r_str   = f"+{reward:.1f}" if reward > 0 else f"{reward:.1f}"
    print(f"{estilo}  {simbolo}  {etiq:<9}{Style.RESET_ALL}  "
          f"ep {ep:03d}  reward {r_str}")


# ─────────────────────────────────────────────────────────────
# 6. BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def entrenar(episodios: int = 100, baud: int = 115200):
    red    = Perceptron(n=4, lr=0.1)
    qtable = QTableRL(n=4)
    historial_rewards: list[float] = []
    aciertos = 0

    # ── Intentar conectar WonderMV ──────────────────────────
    puerto = detectar_puerto_wondermv()
    usar_serial = False
    ser = None

    if puerto:
        try:
            ser = serial.Serial(puerto, baud, timeout=1)
            usar_serial = True
            print(f"✅ WonderMV detectada en {puerto} — modo cámara real")
        except serial.SerialException as e:
            print(f"⚠ No se pudo abrir {puerto}: {e} — modo simulado")
    else:
        print("⚠ WonderMV no encontrada — modo simulado (muestra colores aleatorios)")

    print("\n=== INICIO DE ENTRENAMIENTO ===\n")

    colores_lista = ["rojo", "verde", "amarillo", None]

    for ep in range(1, episodios + 1):

        # ── Leer color (serial real o simulado) ────────────
        if usar_serial:
            try:
                linea = ser.readline().decode("utf-8", errors="ignore")
                print("DEBUG:", repr(linea))
                color_entrada = color_leido(linea)
            except Exception:
                color_entrada = None
        else:
            # Simulación: 80% objetos de color, 20% inactivo
            color_entrada = random.choices(
                ["rojo", "verde", "amarillo", None],
                weights=[30, 30, 30, 10]
            )[0]
            time.sleep(0.05)

        estado = COLOR_ID[color_entrada]

        # ── Elegir acción (Q-table ε-greedy) ───────────────
        accion_id = qtable.elegir_accion(estado)
        color_accion = ID_COLOR[accion_id]

        # ── Calcular reward ─────────────────────────────────
        acierto = (color_accion == color_entrada)
        if acierto:
            reward = round(random.uniform(6.0, 10.0), 1)
            aciertos += 1
        else:
            reward = -round(random.uniform(1.0, 5.0), 1)

        historial_rewards.append(reward)

        # ── Actualizar red y Q-table ────────────────────────
        red.actualizar(estado, estado, reward)     # la red aprende: entrada → misma salida
        qtable.actualizar(estado, accion_id, reward, estado)
        qtable.decaer_epsilon()

        # ── Mostrar en terminal ─────────────────────────────
        desplegar_color(color_accion, acierto, ep, reward)

        # ── Verificar convergencia ──────────────────────────
        if qtable.esta_listo(historial_rewards):
            print(f"\n✅ Modelo listo en episodio {ep}")
            print(f"   Promedio reward (últimos 10): "
                  f"{np.mean(historial_rewards[-10:]):.2f}")
            break

    # ── Resumen final ───────────────────────────────────────
    total_ep = len(historial_rewards)
    print(f"\n{'='*40}")
    print(f"Episodios: {total_ep}")
    print(f"Aciertos : {aciertos} ({aciertos/total_ep*100:.1f}%)")
    print(f"Reward promedio final: {np.mean(historial_rewards[-10:]):.2f}")
    print(f"\nQ-table aprendida:")
    nombres = [ID_COLOR[i] if ID_COLOR[i] is not None else "inactivo" for i in range(4)]
    header = "           " + "  ".join(f"{n:<10}" for n in nombres)
    print(header)
    for s in range(4):
        nombre_s = nombres[s]
        fila = f"{nombre_s:<11}" + "  ".join(f"{v:+.3f}    " for v in qtable.Q[s])
        print(fila)

    if ser:
        ser.close()


# ─────────────────────────────────────────────────────────────
# 7. PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    entrenar(episodios=120, baud=115200)
