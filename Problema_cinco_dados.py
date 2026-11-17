import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Carpetas y parametros
# ---------------------------------------------------------------------
CARPETA = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(CARPETA, "resultados")
os.makedirs(RESULTADOS, exist_ok=True)

# Los cuatro videos mantienen la camara y la escala aproximada de los dados.
AREA_MIN_DADO = 2200  # Area minima observada de la region roja de un dado.
AREA_MAX_DADO = 6000  # Area maxima observada; evita regiones grandes de la mano.
# Rangos medidos sobre las cajas de los dados presentes en los cuatro videos.
ANCHO_MIN_DADO = 48  # Limite inferior del ancho de un dado en pixeles.
ANCHO_MAX_DADO = 100  # Limite superior del ancho por perspectiva y orientacion.
ALTO_MIN_DADO = 48  # Limite inferior del alto en pixeles.
ALTO_MAX_DADO = 105  # Limite superior del alto por perspectiva y orientacion.
PROPORCION_MIN_DADO = 0.68  # La rotacion permite cajas algo mas altas que anchas.
PROPORCION_MAX_DADO = 1.35  # La rotacion permite cajas algo mas anchas que altas.
FRAMES_CONSECUTIVOS = 5  # Confirma reposo durante 5 comparaciones (1/6 s a 30 FPS).
DISTANCIA_MAXIMA = 7  # Pixeles de desplazamiento maximo del centro entre frames.
DIFERENCIA_MAXIMA = 15  # Diferencia media en gris admitida en cada dado (0..255).

# Se unen interrupciones de hasta 2 frames (0,067 s a 30 FPS):
# este margen cubre fluctuaciones breves de iluminacion/contornos.
HUECO_MAXIMO = 2
FACTOR_UNION_TRAMOS = 2  # Entre tramos se permiten 14 px (2 x 7 px)
# respecto de la posicion de referencia para absorber pequenas oscilaciones.

# Espacio HSV del paño verde, identificado en los cuatro videos.
# H=34..100 cubre verdes/cianes; S>=50 evita el blanco/gris de la mesa.
VERDE_MIN = (34, 50, 35)
VERDE_MAX = (100, 255, 255)

# El rojo se separa en dos intervalos porque H es circular en HSV.
# S>=150 cae en el valle del histograma: la piel queda por debajo
# (S ~ 30-120) y los dados por encima (S ~ 180-230).
# V>=50 evita sombras oscuras.
ROJO_1_MIN, ROJO_1_MAX = (0, 150, 50), (12, 255, 255)
ROJO_2_MIN, ROJO_2_MAX = (170, 150, 50), (179, 255, 255)

# S<=100 excluye los reflejos rosados de las caras inferiores.
# V>=170 conserva los puntos claros de la cara superior.
SATURACION_MAX_PUNTO = 100
VALOR_MIN_PUNTO = 170

# El punto blanco visible ocupa normalmente entre 1% y 2% del
# bounding box del dado. Se admite 0,8%-4% por cambios de escala/sombra.
AREA_PUNTO_MIN_REL = 0.008
AREA_PUNTO_MAX_REL = 0.04

# Factor de forma 4*pi*A/P^2; un circulo ideal vale 1.
# Se admite 0,55 por discretizacion y perspectiva de las caras inclinadas.
FACTOR_FORMA_MIN = 0.55
PROPORCION_MIN_PUNTO = 0.55  # Tolera elipses por perspectiva y pixelado.
PROPORCION_MAX_PUNTO = 1.6  # Descarta regiones blancas alargadas.
TAM_EE_CLAUSURA = 5  # EE 5x5: une cortes pequenos en las regiones rojas.
TAM_EE_EROSION = 3  # EE 3x3: separa el interior del borde del dado.
FRAME_MANO = 90  # En tirada 1 se ve la mano entrando al paño.

# Visualizaciones no bloqueantes hasta el grafico final.
# Poner False si se desea generar unicamente los archivos sin ventanas.
MOSTRAR_FIGURAS = True


# ---------------------------------------------------------------------
# 2. Segmentar los dados por color (HSV) y filtrar por geometria
# ---------------------------------------------------------------------
# La region del paño delimita la zona de juego y excluye la madera.
# Como la camara permanece fija, se calcula una sola vez por video.
def obtener_mascara_pano(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, VERDE_MIN, VERDE_MAX)

    # Conservamos la mayor componente conectada (paño verde).
    cantidad, etiquetas, datos, _ = cv2.connectedComponentsWithStats(
        verde, connectivity=8
    )
    mascara = np.zeros(verde.shape, dtype=np.uint8)
    if cantidad <= 1:
        return mascara

    mayor = 1 + np.argmax(datos[1:, cv2.CC_STAT_AREA])
    mascara[etiquetas == mayor] = 255

    # Rellenamos agujeros (incluidos los dejados por los dados) mediante
    # el contorno externo, tecnica de segmentacion/morfologia de clase.
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    mascara_rellena = np.zeros(mascara.shape, dtype=np.uint8)
    cv2.drawContours(mascara_rellena, contornos, -1, 255, -1)
    return mascara_rellena


def detectar_dados(frame, mascara_pano=None):
    # Se procesa el frame completo, de modo que la mascara conserve
    # las dimensiones del video y pueda compararse con las figuras.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_1 = cv2.inRange(hsv, ROJO_1_MIN, ROJO_1_MAX)
    mascara_2 = cv2.inRange(hsv, ROJO_2_MIN, ROJO_2_MAX)
    mascara_rojo = cv2.bitwise_or(mascara_1, mascara_2)

    if mascara_pano is not None:
        mascara_rojo = cv2.bitwise_and(mascara_rojo, mascara_pano)

    # Clausura con EE 5x5 para unir cortes pequenos del borde rojo.
    # No rellena los puntos blancos (~10-13 px de diametro); tampoco
    # es necesario porque cada dado se delimita por su bounding box.
    ee = cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (TAM_EE_CLAUSURA, TAM_EE_CLAUSURA))
    mascara_rojo = cv2.morphologyEx(mascara_rojo, cv2.MORPH_CLOSE, ee)

    # Cada componente conectada es una candidata a dado.
    cantidad, etiquetas, datos, centros = cv2.connectedComponentsWithStats(
        mascara_rojo, connectivity=8
    )

    dados = []
    for i in range(1, cantidad):
        x, y, w, h, area = datos[i]
        # El area, las dimensiones y la proporcion filtran regiones
        # rojas que no corresponden a un dado.
        if (AREA_MIN_DADO <= area <= AREA_MAX_DADO
                and ANCHO_MIN_DADO <= w <= ANCHO_MAX_DADO
                and ALTO_MIN_DADO <= h <= ALTO_MAX_DADO
                and PROPORCION_MIN_DADO <= w / h <= PROPORCION_MAX_DADO):
            dados.append((int(x), int(y), int(w), int(h)))

    # Se asignan IDs inicialmente de arriba hacia abajo; despues se
    # conservan por proximidad de centroides dentro de cada reposo.
    dados.sort(key=lambda d: (d[1] + d[3] / 2, d[0]))
    return dados, mascara_rojo


# ---------------------------------------------------------------------
# 3. Segmentar y contar los puntos blancos
# ---------------------------------------------------------------------
def contar_puntos(frame, dado, devolver_aceptados=False):
    x, y, w, h = dado
    recorte = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)

    # El blanco de los puntos presenta valor alto y saturacion baja.
    mascara = cv2.inRange(hsv, (0, 0, VALOR_MIN_PUNTO),
                          (179, SATURACION_MAX_PUNTO, 255))

    # La capsula convexa erosionada delimita el interior del dado.
    # Esto excluye el fondo y conserva los puntos cercanos a las esquinas.
    rojo_1 = cv2.inRange(hsv, ROJO_1_MIN, ROJO_1_MAX)
    rojo_2 = cv2.inRange(hsv, ROJO_2_MIN, ROJO_2_MAX)
    rojo = cv2.bitwise_or(rojo_1, rojo_2)
    contornos_rojos, _ = cv2.findContours(rojo, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    if contornos_rojos:
        contorno = max(contornos_rojos, key=cv2.contourArea)
        casco = cv2.convexHull(contorno)
        interior = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(interior, [casco], -1, 255, -1)
        # EE 3x3: descarta unicamente el contorno externo del dado.
        ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (TAM_EE_EROSION, TAM_EE_EROSION))
        interior = cv2.erode(interior, ee, iterations=1)
        mascara = cv2.bitwise_and(mascara, interior)
    else:
        mascara[:, :] = 0

    # Area relativa, proporcion y factor de forma (descriptores U7).
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    puntos = []
    mascara_aceptados = np.zeros(mascara.shape, dtype=np.uint8)
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue
        factor_forma = 4 * np.pi * area / perimetro**2
        px, py, pw, ph = cv2.boundingRect(contorno)
        area_relativa = area / (w * h)
        if (AREA_PUNTO_MIN_REL <= area_relativa <= AREA_PUNTO_MAX_REL
                and factor_forma >= FACTOR_FORMA_MIN
                and PROPORCION_MIN_PUNTO <= pw / ph <= PROPORCION_MAX_PUNTO):
            puntos.append((px + pw / 2, py + ph / 2))
            # La mascara de aceptados se usa solo para visualizar el conteo.
            cv2.drawContours(mascara_aceptados, [contorno], -1, 255, -1)

    if devolver_aceptados:
        return len(puntos), mascara, mascara_aceptados
    return len(puntos), mascara


# ---------------------------------------------------------------------
# 4. Comparar frames consecutivos para determinar reposo
# ---------------------------------------------------------------------
def desplazamiento_maximo(dados, dados_anteriores):
    """Mayor distancia de centroides; NaN si no se detectan cinco dados."""
    if len(dados) != 5 or len(dados_anteriores) != 5:
        return np.nan
    distancias = []
    for dado, anterior in zip(dados, dados_anteriores):
        x, y, w, h = dado
        xa, ya, wa, ha = anterior
        cx, cy = x + w / 2, y + h / 2
        cxa, cya = xa + wa / 2, ya + ha / 2
        distancias.append(np.sqrt((cx - cxa)**2 + (cy - cya)**2))
    return max(distancias)


def estan_quietos(dados, valores, dados_anteriores, valores_anteriores,
                  gris, gris_anterior):
    if (len(dados) != 5 or len(dados_anteriores) != 5
            or gris_anterior is None):
        return False

    # Cada cara valida tiene entre 1 y 6 puntos. Un valor fuera de ese
    # rango indica giro, desenfoque u oclusion: no puede informarse
    # con seguridad el resultado durante ese frame.
    if any(valor < 1 or valor > 6 for valor in valores):
        return False

    # El reposo depende del desplazamiento y del cambio de intensidad.
    # Pequenas variaciones del conteo se estabilizan despues con la moda.
    if desplazamiento_maximo(dados, dados_anteriores) > DISTANCIA_MAXIMA:
        return False

    for x, y, w, h in dados:
        actual = gris[y:y+h, x:x+w]
        anterior = gris_anterior[y:y+h, x:x+w]
        diferencia = cv2.absdiff(actual, anterior)
        if np.mean(diferencia) > DIFERENCIA_MAXIMA:
            return False
    return True


# ---------------------------------------------------------------------
# 5. Detectar intervalos de reposo (primera lectura del video)
# ---------------------------------------------------------------------
def buscar_reposo(entrada):
    # Primera pasada: se registran dados, puntos y comparaciones.
    cap = cv2.VideoCapture(entrada)
    if not cap.isOpened():
        return [], [], None
    ok, primero = cap.read()
    if not ok:
        cap.release()
        return [], [], None

    mascara_pano = obtener_mascara_pano(primero)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    registros = []
    dados_anteriores = []
    valores_anteriores = []
    gris_anterior = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dados, mascara_rojo = detectar_dados(frame, mascara_pano)
        valores = [contar_puntos(frame, dado)[0] for dado in dados]
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        distancia = desplazamiento_maximo(dados, dados_anteriores)
        quietos = estan_quietos(dados, valores, dados_anteriores,
                                valores_anteriores, gris, gris_anterior)
        registros.append({"dados": dados, "valores": valores,
                          "quietos": quietos, "distancia": distancia})
        dados_anteriores = dados
        valores_anteriores = valores
        gris_anterior = gris
    cap.release()

    # Minimo de FRAMES_CONSECUTIVOS comparaciones quietas.
    # Como la primera comparacion corresponde al paso i-1 --> i,
    # el tramo incluye i-1 para no perder el primer frame del reposo.
    intervalos = []
    inicio = None
    for i in range(len(registros) + 1):
        quietos = registros[i]["quietos"] if i < len(registros) else False
        if quietos and inicio is None:
            inicio = i
        elif not quietos and inicio is not None:
            if i - inicio >= FRAMES_CONSECUTIVOS:
                intervalos.append((inicio - 1, i - 1))
            inicio = None

    # Se unen solo huecos de 1 o 2 frames con cinco dados validos,
    # y que permanezcan cerca de la configuracion estable anterior.
    # Evitamos prolongar el reposo cuando entra una mano y retira dados.
    unidos = []
    for inicio, fin in intervalos:
        if unidos:
            ant_inicio, ant_fin = unidos[-1]
            hueco = inicio - ant_fin - 1
            if 0 <= hueco <= HUECO_MAXIMO:
                referencia = registros[ant_fin]["dados"]
                cerca = True
                for j in range(ant_fin + 1, inicio + 1):
                    reg = registros[j]
                    if (len(reg["dados"]) != 5
                            or any(v < 1 or v > 6 for v in reg["valores"])):
                        cerca = False
                        break
                    # Se tolera FACTOR_UNION_TRAMOS veces la distancia
                    # normal entre tramos, siempre respecto de la referencia.
                    if (desplazamiento_maximo(reg["dados"], referencia)
                            > FACTOR_UNION_TRAMOS * DISTANCIA_MAXIMA):
                        cerca = False
                        break
                if cerca:
                    unidos[-1] = (ant_inicio, fin)
                    continue
        unidos.append((inicio, fin))
    return registros, unidos, mascara_pano


# Se asignan los IDs por altura al iniciar el reposo y se conservan
# mediante la cercania al centroide de referencia.
def ordenar_por_id(dados, referencia):
    disponibles = list(dados)
    ordenados = []
    for ref in referencia:
        if not disponibles:
            return []
        rx, ry, rw, rh = ref
        rcx, rcy = rx + rw / 2, ry + rh / 2
        indice = min(range(len(disponibles)), key=lambda i:
                     (disponibles[i][0] + disponibles[i][2] / 2 - rcx)**2
                     + (disponibles[i][1] + disponibles[i][3] / 2 - rcy)**2)
        ordenados.append(disponibles.pop(indice))
    return ordenados


# Moda de los valores de cada ID a lo largo de un intervalo.
def valores_estables(registros, inicio, fin):
    referencia = registros[inicio]["dados"]
    if len(referencia) != 5:
        return None, None, {}

    # Guardamos las cajas alineadas por ID y los conteos de cada frame.
    cajas_por_frame = {}
    valores_por_id = [[] for _ in range(5)]
    for j in range(inicio, fin + 1):
        dados = registros[j]["dados"]
        if len(dados) != 5:
            continue
        ordenados = ordenar_por_id(dados, referencia)
        if len(ordenados) != 5:
            continue
        valores = registros[j]["valores"]
        mapeo = {tuple(d): v for d, v in zip(dados, valores)}
        valores_ordenados = [mapeo[tuple(d)] for d in ordenados]
        cajas_por_frame[j] = ordenados
        for i, valor in enumerate(valores_ordenados):
            if 1 <= valor <= 6:
                valores_por_id[i].append(valor)

    modas = []
    porcentajes = []
    for lista in valores_por_id:
        if not lista:
            return None, None, cajas_por_frame
        frecuencias = np.bincount(lista, minlength=7)
        moda = int(np.argmax(frecuencias[1:]) + 1)
        modas.append(moda)
        porcentajes.append(100 * frecuencias[moda] / len(lista))
    return modas, porcentajes, cajas_por_frame


# ---------------------------------------------------------------------
# 6. Dibujar etiquetas y guardar etapas del procesamiento
# ---------------------------------------------------------------------
def dibujar_dados(frame, dados, valores):
    for i, ((x, y, w, h), valor) in enumerate(zip(dados, valores), start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        texto = f"D{i}-{valor}"
        posicion = (x, max(35, y - 12))
        cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 5)
        cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2)
    return frame


def guardar_etapas(numero, original, mascara_rojo, dados, resultado):
    # Ambas mascaras tienen las mismas dimensiones que el frame.
    puntos_blancos = np.zeros(mascara_rojo.shape, dtype=np.uint8)
    for dado in dados:
        x, y, w, h = dado
        _, mascara = contar_puntos(original, dado)
        puntos_blancos[y:y+h, x:x+w] = mascara

    fig, ejes = plt.subplots(2, 2, figsize=(10, 13))
    ejes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ejes[0, 0].set_title("Frame original")
    ejes[0, 1].imshow(mascara_rojo, cmap="gray")
    ejes[0, 1].set_title("Mascara de dados rojos (dentro del paño)")
    ejes[1, 0].imshow(puntos_blancos, cmap="gray")
    ejes[1, 0].set_title("Mascara de puntos blancos")
    ejes[1, 1].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    ejes[1, 1].set_title("Dados con ID y valor estable")
    for eje in ejes.flat:
        eje.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTADOS, f"etapas_tirada_{numero}.png"), dpi=120)
    if MOSTRAR_FIGURAS:
        plt.show(block=False)
    else:
        plt.close(fig)


def guardar_detalle_puntos(numero, original, dados):
    """Mostrar por dado el recorte, los blancos candidatos y los aceptados."""
    fig, ejes = plt.subplots(3, 5, figsize=(16, 10))
    for i, dado in enumerate(dados):
        x, y, w, h = dado
        recorte = original[y:y+h, x:x+w]
        cantidad, mascara, aceptados = contar_puntos(
            original, dado, devolver_aceptados=True
        )
        ejes[0, i].imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        ejes[0, i].set_title(f"Dado {i + 1}: {cantidad} punto{'s' if cantidad != 1 else ''}")
        ejes[1, i].imshow(mascara, cmap="gray", vmin=0, vmax=255)
        ejes[2, i].imshow(aceptados, cmap="gray", vmin=0, vmax=255)
        for fila in range(3):
            ejes[fila, i].axis("off")

    fig.suptitle(f"Tirada {numero} - conteo en el primer frame de reposo")
    # Los nombres de las filas quedan fuera de los ejes de imagen.
    fig.tight_layout(rect=(0.10, 0, 1, 0.96))
    fig.text(0.015, 0.80, "Recorte en color", rotation=90, va="center")
    fig.text(0.015, 0.49, "Mascara de blancos", rotation=90, va="center")
    fig.text(0.015, 0.19, "Puntos aceptados", rotation=90, va="center")
    fig.savefig(os.path.join(RESULTADOS, "puntos_tirada_1.png"), dpi=130)
    if MOSTRAR_FIGURAS:
        plt.show(block=False)
    else:
        plt.close(fig)


def guardar_histograma_s(entrada, mascara_pano):
    # En la tirada 1, frame 90, se ve la mano entrando desde la izquierda.
    # Se analiza la saturacion de los pixeles de tono rojo dentro
    # del paño, para justificar el limite S=150.
    cap = cv2.VideoCapture(entrada)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_MANO)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    rojos = ((H <= ROJO_1_MAX[0]) | (H >= ROJO_2_MIN[0]))
    seleccion = rojos & (mascara_pano > 0) & (V >= ROJO_1_MIN[2])
    sat = S[seleccion]

    fig, eje = plt.subplots(figsize=(9, 4))
    # 50 barras equivalen a intervalos de ~5 niveles de S; permiten
    # ver los grupos de baja y alta saturacion sin perder el detalle.
    eje.hist(sat, bins=50, range=(0, 255), color="firebrick")
    eje.axvline(ROJO_1_MIN[1], color="black", linestyle="--",
                label=f"Umbral de dado: S >= {ROJO_1_MIN[1]}")
    eje.set(title=f"Tirada 1 - saturacion de tonos rojos (frame {FRAME_MANO})",
            xlabel="Saturacion S (0 a 255)", ylabel="Cantidad de pixeles")
    eje.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTADOS, "histograma_s_tirada_1.png"), dpi=130)
    if MOSTRAR_FIGURAS:
        plt.show(block=False)
    else:
        plt.close(fig)


def guardar_desplazamientos(desplazamientos):
    # Grafico conjunto de los cuatro videos y limite horizontal.
    fig, ejes = plt.subplots(4, 1, figsize=(11, 9))
    for i, eje in enumerate(ejes, start=1):
        serie = np.array(desplazamientos[i])
        eje.plot(np.arange(len(serie)), serie, linewidth=1)
        eje.axhline(DISTANCIA_MAXIMA, color="red", linestyle="--",
                    label=f"Limite de distancia = {DISTANCIA_MAXIMA} px")
        # El eje 0..30 px muestra en detalle los desplazamientos cercanos
        # al umbral 7 px; las distancias mayores solo se recortan en el dibujo.
        eje.set_ylim(0, 30)
        eje.set(title=f"Tirada {i} - vista ampliada (0-30 px)",
                ylabel="Desplazamiento max. (px)")
        eje.legend(loc="upper right")
    ejes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTADOS, "desplazamientos_cuatro_tiradas.png"), dpi=130)
    if MOSTRAR_FIGURAS:
        plt.show()  # Ultima visualizacion bloqueante.
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# 7. Grabar cada video con etiquetas durante los intervalos de reposo
# ---------------------------------------------------------------------
def procesar_video(numero):
    entrada = os.path.join(CARPETA, f"tirada_{numero}.mp4")
    salida = os.path.join(RESULTADOS, f"tirada_{numero}_resultado.mp4")

    cap = cv2.VideoCapture(entrada)
    if not cap.isOpened():
        print(f"No se pudo abrir: {entrada}")
        return None

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        print(f"FPS incorrecto en {entrada}")
        return None

    registros, intervalos, mascara_pano = buscar_reposo(entrada)
    if not registros:
        return None

    # Se asigna a todos los frames de reposo el valor modal de cada dado.
    etiquetas_por_frame = {}
    resumen = None
    for inicio, fin in intervalos:
        valores, porcentajes, cajas = valores_estables(registros, inicio, fin)
        if valores is None:
            continue
        print(f"\nTIRADA {numero} - reposo: frames {inicio} a {fin} "
              f"({inicio/fps:.2f} a {fin/fps:.2f} s)")
        for i, (v, pct) in enumerate(zip(valores, porcentajes), start=1):
            print(f"  D{i}: {v} (coincidencia {pct:.1f}%)")
        print(f"  Suma: {sum(valores)}")
        # Se conserva el primer intervalo valido para el resumen final.
        if resumen is None:
            resumen = {"inicio": inicio, "fin": fin,
                       "valores": valores, "suma": sum(valores)}
        for k, dados in cajas.items():
            etiquetas_por_frame[k] = (dados, valores)

    if not etiquetas_por_frame:
        print(f"\nTIRADA {numero}: no se detecto un intervalo de reposo")

    # Segunda pasada: grabar como en el ejemplo de la catedra,
    # con VideoCapture, VideoWriter, rectangle, putText y release.
    cap = cv2.VideoCapture(entrada)
    out = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (ancho, alto))
    if not out.isOpened():
        print(f"No se pudo crear el video: {salida}")
        cap.release()
        return None

    numero_frame = 0
    ejemplo_guardado = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if numero_frame in etiquetas_por_frame:
            dados, valores = etiquetas_por_frame[numero_frame]
            if not ejemplo_guardado:
                original = frame.copy()
            frame = dibujar_dados(frame, dados, valores)
            if not ejemplo_guardado:
                _, mascara_rojo = detectar_dados(original, mascara_pano)
                guardar_etapas(numero, original, mascara_rojo, dados, frame)
                if numero == 1:
                    guardar_detalle_puntos(numero, original, dados)
                cv2.imwrite(os.path.join(RESULTADOS,
                            f"ejemplo_tirada_{numero}.jpg"), frame)
                ejemplo_guardado = True
        out.write(frame)
        numero_frame += 1

    cap.release()
    out.release()
    print(f"  Video {numero}: {len(etiquetas_por_frame)} frames etiquetados")
    print(f"  Salida: {salida}")
    if resumen is not None:
        resumen["etiquetados"] = len(etiquetas_por_frame)
        resumen["fps"] = fps
    return {"distancias": [r["distancia"] for r in registros],
            "resumen": resumen, "mascara_pano": mascara_pano}


# ---------------------------------------------------------------------
# 8. Ejecutar las cuatro tiradas
# ---------------------------------------------------------------------
if __name__ == "__main__":
    desplazamientos = {}
    resumen_final = []
    mascara_pano_1 = None
    for numero in range(1, 5):
        resultado = procesar_video(numero)
        if resultado is not None:
            desplazamientos[numero] = resultado["distancias"]
            if resultado["resumen"] is not None:
                resumen_final.append((numero, resultado["resumen"]))
            if numero == 1:
                mascara_pano_1 = resultado["mascara_pano"]

    # Resumen de los resultados obtenidos con las cuatro tiradas.
    print("\nRESUMEN FINAL")
    print("Tirada | Reposo (frames) | Reposo (s) | D1 D2 D3 D4 D5 | Suma | Etiquetas")
    for numero, dato in resumen_final:
        inicio, fin, fps = dato["inicio"], dato["fin"], dato["fps"]
        valores = " ".join(str(v) for v in dato["valores"])
        print(f"{numero:>6} | {inicio:>3}-{fin:<3}         | "
              f"{inicio/fps:.2f}-{fin/fps:.2f} | {valores:<14} | "
              f"{dato['suma']:>4} | {dato['etiquetados']:>9}")

    # Las figuras de etapas ya se guardaron durante cada procesamiento.
    # Se muestra el histograma de la primera tirada y, al final,
    # el grafico conjunto de desplazamientos (ultima ventana bloqueante).
    if mascara_pano_1 is not None:
        guardar_histograma_s(os.path.join(CARPETA, "tirada_1.mp4"), mascara_pano_1)
    if all(desplazamientos.get(i) for i in range(1, 5)):
        guardar_desplazamientos(desplazamientos)
