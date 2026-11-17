# TP3 - Procesamiento de Imágenes: Detección y Conteo de Dados en Video

Resolución del Trabajo Práctico N° 3 de la materia **Procesamiento de Imágenes** (Licenciatura en Ciencia de Datos - Universidad Austral, 2026).

Este repositorio contiene la implementación en Python para la detección, seguimiento del desplazamiento y conteo automático de puntos de cinco dados a partir de secuencias de video.

## Integrantes
* Justo Borrás
* Santino Almiron Nanni
* Wenceslao Tejerina

---

## Estructura del Repositorio

* `Problema_cinco_dados.py`: Script principal que procesa los videos de las tiradas, detecta los dados en reposo, contabiliza el valor de sus caras superiores y genera las salidas gráficas y audiovisuales.
* `tirada_1.mp4` a `tirada_4.mp4`: Videos originales de entrada correspondientes a las cuatro tiradas de cinco dados.
* `requirements.txt`: Dependencias de Python necesarias para la ejecución del proyecto.
* `resultados/`: Carpeta que almacena los videos procesados (`tirada_1_resultado.mp4` a `tirada_4_resultado.mp4`) y las figuras de las etapas intermedias y análisis (`etapas_tirada_1.png` a `etapas_tirada_4.png`, `desplazamientos_cuatro_tiradas.png`, `histograma_s_tirada_1.png`, `puntos_tirada_1.png` y ejemplos de fotogramas).
* `Informe_TP3.pdf`: Informe técnico en formato PDF con la descripción del problema, análisis metodológico, capturas de pasos intermedios y conclusiones.

---

## Instalación y Configuración del Entorno

Se recomienda utilizar un entorno virtual de Python 3.x para instalar las librerías requeridas sin afectar la configuración global del sistema.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/wenceslao015/TP_final_pdi2026.git](https://github.com/wenceslao015/TP_final_pdi2026.git)
   cd TP_final_pdi2026
   ```

2. **Crear el entorno virtual:**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual:**
   * **Windows (PowerShell o CMD):**
```bash
venv\Scripts\activate
```
   * **Linux / macOS / WSL:**
```bash
source venv/bin/activate
```

4. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución del Script

Con el entorno virtual activado y ubicado en el directorio raíz del proyecto (junto a los archivos `tirada_1.mp4` a `tirada_4.mp4`), ejecuta el siguiente comando en la terminal:

```bash
python Problema_cinco_dados.py
```

### Resultados Generados
Durante la ejecución, el script procesará automáticamente las cuatro secuencias de video y guardará todos los resultados en la carpeta `resultados/`:
* **Videos procesados:** `tirada_1_resultado.mp4` a `tirada_4_resultado.mp4` con las detecciones, el seguimiento y el conteo superpuestos.
* **Capturas intermedias y métricas:** Imágenes `.png` y `.jpg` con las etapas morfológicas, histogramas de segmentación, detección de puntos y gráficos de desplazamiento de los dados.