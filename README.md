# Realidade Aumentada com ArUco

**Integrantes:**

[Enrico Cuono Alves Pereira - 10402875](https://github.com/Enrico258)

[Gabriel Mason Guerino - 10409928](https://github.com/GabrielMasonGuerino)

[Eduardo Honorio Friaça - 10408959](https://github.com/EduardoFriaca)

# O que é Realidade Aumentada (RA)

A Realidade Aumentada (RA) combina o mundo real com elementos virtuais.

Ela adiciona objetos 3D, textos ou imagens sobre o ambiente real em tempo real.

Exemplos: filtros do Instagram e jogos como Pokémon GO.

Diferente da Realidade Virtual, a RA não substitui o mundo real — apenas o complementa.

<img width="480" height="365" alt="image" src="https://github.com/user-attachments/assets/4fe93825-72c0-42a9-b11b-cd458375b8fc" />

Fonte da imagem: [Niantic Help Center](https://niantic.helpshift.com/hc/pt/6-pokemon-go/faq/28-catching-pokemon-in-ar-mode-1712012768/)

# Marcadores ArUco

Os marcadores ArUco são padrões quadrados em preto e branco usados para detectar posições e orientações no espaço.

Cada marcador tem um ID único e faz parte de um dicionário, como DICT_4X4_50.

Eles ajudam a “ancorar” objetos virtuais no mundo real.

<img width="850" height="635" alt="image" src="https://github.com/user-attachments/assets/98749c4f-22e5-4527-98db-cc34b33ec380" />

Fonte da imagem: [ResearchGate](https://www.researchgate.net/figure/DICT-6X6-250-dictionary-of-ArUco-markers_fig9_366600173)

# Detecção dos Cantos do Marcador

Depois de detectar o marcador, o algoritmo encontra os quatro cantos dele.

Esses pontos são usados pra calcular a posição e a orientação da câmera em relação ao marcador.

<img width="700" height="518" alt="image" src="https://github.com/user-attachments/assets/a2e3fbaf-8a50-4036-8f38-166b6d2761b5" />

Fonte da imagem: [PyImageSearch](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)



## 4️Matriz Intrínseca da Câmera

A **matriz intrínseca** descreve as características internas da câmera — ou seja, como ela transforma pontos tridimensionais (3D) do mundo real em coordenadas bidimensionais (2D) na imagem capturada.
Ela é baseada no **modelo pinhole**, um modelo ideal que considera um único ponto de projeção, sem lentes.

### 📐 Modelo de Câmera Pinhole

O modelo **pinhole** assume que todos os raios de luz passam por um único ponto e formam a imagem no plano oposto.
Apesar de simples, esse modelo representa bem o comportamento óptico das câmeras digitais.

<img width="600" alt="pinhole" src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Pinhole_camera_model.svg">

**Figura:** Representação do modelo de câmera pinhole.
*Fonte: Wikimedia Commons*

---

### ⚙️ Parâmetros Intrínsecos

Os **parâmetros intrínsecos** são obtidos através de calibração e determinam como a câmera projeta o mundo real na imagem:

* **Distância focal (f)** → define o campo de visão da câmera
* **Ponto principal (cₓ, cᵧ)** → centro óptico da imagem
* **Coeficientes de distorção** → corrigem deformações da lente (radiais e tangenciais)

A matriz intrínseca tem a forma:

[
K =
\begin{bmatrix}
f_x & 0 & c_x \
0 & f_y & c_y \
0 & 0 & 1
\end{bmatrix}
]

---

### 🔧 Calibração Real vs. Aproximação

Uma **calibração real** é feita com imagens de um padrão conhecido (ex: tabuleiro de xadrez), posicionadas em diferentes ângulos.
O OpenCV detecta os cantos e calcula automaticamente a matriz **K** e os coeficientes de distorção.

Quando não há tempo para calibrar, é possível usar uma **aproximação** — estimando os parâmetros intrínsecos com base na resolução e no campo de visão.

Exemplo em Python:

```python
def estimate_intrinsics_approx(w, h, fov_deg=60):
    f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((1, 5), dtype=np.float32)
    return K, dist
```

Essa função fornece uma **estimativa inicial aceitável** para testes de realidade aumentada, sem necessidade de calibração precisa.

<img width="700" alt="calibration" src="https://docs.opencv.org/4.x/calibration_chessboard.png">

**Figura:** Processo de calibração de câmera com padrão de xadrez.
*Fonte: OpenCV Documentation*

---

## Pose Estimation — PnP (Perspective-n-Points)

A **estimativa de pose (Pose Estimation)** define a **posição e a orientação da câmera** em relação a um marcador ou objeto conhecido.
É a etapa que permite projetar corretamente objetos 3D no ambiente real.

### 📏 Conceito

O método **PnP (Perspective-n-Points)** utiliza correspondências entre:

* 4 pontos **3D conhecidos** (por exemplo, os cantos do marcador)
* 4 pontos **2D detectados** na imagem

A partir disso, o algoritmo calcula:

* **rvec (rotation vector)** → rotação da câmera
* **tvec (translation vector)** → posição da câmera

Esses vetores permitem desenhar eixos e cubos 3D alinhados com o marcador.

<img width="700" alt="pnp" src="https://docs.opencv.org/4.x/pnp_pose_estimation.png">

**Figura:** Relação entre pontos 3D e 2D na estimação de pose.
*Fonte: OpenCV Documentation*

---

### 🧮 Cálculo no OpenCV

O cálculo da pose é feito com a função `cv2.solvePnP()`:

```python
success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
```

**Parâmetros:**

* `objp` → coordenadas 3D conhecidas (ex: cantos do marcador)
* `imgp` → pontos 2D detectados na imagem
* `K` → matriz intrínseca da câmera
* `dist` → coeficientes de distorção

Se `success` for verdadeiro, o OpenCV retorna `rvec` e `tvec`, que podem ser usados para desenhar objetos 3D com precisão.

---

### ⚙️ Algoritmos do PnP

| **Algoritmo**          | **Características**                           |
| ---------------------- | --------------------------------------------- |
| `SOLVEPNP_ITERATIVE`   | Método clássico, robusto e preciso            |
| `SOLVEPNP_P3P`         | Rápido, ideal para 3 pontos                   |
| `SOLVEPNP_IPPE_SQUARE` | Recomendado para marcadores quadrados (ArUco) |
| `SOLVEPNP_AP3P`        | Variante aprimorada para pequenas distâncias  |

Na prática, o mais usado é o `SOLVEPNP_IPPE_SQUARE`, pois fornece bons resultados com marcadores ArUco.
Caso falhe, o código pode tentar novamente com o método padrão `SOLVEPNP_ITERATIVE`.

---

### 🎯 Visualização da Pose

Após calcular `rvec` e `tvec`, o sistema desenha os eixos **(X, Y, Z)** e o **cubo 3D** sobre o marcador, mostrando a orientação e posição no espaço.

<img width="640" height="480" alt="aruco_axes" src="https://github.com/user-attachments/assets/f49f1737-5dac-4d46-ba16-8168a5bfd60f">

**Figura:** Eixos X (vermelho), Y (verde) e Z (azul) desenhados com base nos vetores de pose.
*Fonte: OpenCV Documentation*


# Projeção de Objetos 3D:

A **projeção** é o processo de converter pontos do mundo real (3D) para posições no plano da imagem (2D).

A conversão utiliza os parâmetros da câmera (intrínsecos e extrínsecos), obtidos normalmente no processo de calibração.

Dentro da biblioteca openCV há a função projectPoints(), que recebe pontos 3D do objeto, posição e orientação da câmera, parâmetros intrínsecos e coeficientes de distorção. Então retornando as coordenadas 2D projetadas na imagem.

**Desenho de eixos (X, Y, Z):**
Os eixos ajudam a visualizar a orientação do objeto no espaço

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/f49f1737-5dac-4d46-ba16-8168a5bfd60f" />

Fonte da imagem: [OpenCV Documentation](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)





Eles mostram como o sistema de coordenadas do marcados está posicionado. Como é possível ver na imagem, os eixos se diferem, pois os marcadores não estão orientados igualmente.

**Desenho do cubo:**
1. Definir os 8 vértices de um cubo em coordenadas 3D
2. Projetar esses pontos com a função projectPoints()
3. Conectar as arestas com linhas na imagem

# Condições para Rastreamento Estável:

- Marcador grande o suficiente: quanto maior o marcador mais pixels disponíveis, menos ruído e mais precisão.
- Iluminação adequada: aumenta o contraste e facilita no reconhecimento dos padrões ArUco
- Foco e resolução da câmera: ajuda na precisão do cubo
- Calibração para melhor precisão

<img width="952" height="698" alt="image" src="https://github.com/user-attachments/assets/09d670f6-263e-4d6f-9b84-a8eb8d49740d" />


# Interação em Tempo Real:
Para uma iteração ao vivo com o objeto permitindo que ele se mova constantemente conforme a câmera ou o marcador se movem é necessário definir um loop de aquisição de imagem. Como será mostrado na implementação, esse loop está presente na main:

```bash
while True:
        ok, frame = cap.read() #Entrada de dados em tempo real
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Conversão para cinza
        corners, ids, _ = detect(gray, dictionary) # Identifica marcadores e seus vértices

        if ids is not None and len(ids) > 0:
            # desenha contornos
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c in corners:
                c = c.reshape(4, 2).astype(np.float32)
                imgp = c
                success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not success:
                    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
                if success: # Desenha o cubo e os eixos caso o PnP resulte em sucesso
                    draw_axes(frame, K, dist, rvec, tvec, axis_len=args.axis_scale)
                    draw_cube(frame, K, dist, rvec, tvec, side=args.cube_size)

        
        cv2.imshow("AR ArUco - OpenCV", frame) # Atualiza a visualização
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
```

# Implementação prática:

### 1. Importação das bibliotecas

```bash
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path
import cv2
import numpy as np
```

### 2. Mapeamento dos marcadores ArUcos

```bash
def get_aruco_dict(dict_name: str):
    """Resolve o dicionário ArUco a partir de uma string amigável."""
    name = dict_name.strip().upper()
    mapping = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    if name not in mapping:
        raise ValueError(f"Dicionário '{dict_name}' não suportado.")
    return cv2.aruco.getPredefinedDictionary(mapping[name])

```

### 3. Criação do detector de marcadores ArUco

```bash
def build_detector():
    aruco_params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters') else cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, 'ArucoDetector'):
        def detect(frame, dictionary):
            detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
            corners, ids, rejected = detector.detectMarkers(frame)
            return corners, ids, rejected
        return detect
    else:
        def detect(frame, dictionary):
            return cv2.aruco.detectMarkers(frame, dictionary, parameters=aruco_params)
        return detect
```

### 4. Estimador de calibração de webcam

```bash
def estimate_intrinsics_approx(w: int, h: int, fov_deg: float = 60.0):
    f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((1, 5), dtype=np.float32)
    return K, dist
```

### 5. Desenha os eixos

```bash
def draw_axes(img, K, dist, rvec, tvec, axis_len=0.03):
    # Eixos X (vermelho), Y (verde), Z (azul) — cores padrão do OpenCV
    axis = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    p0, px, py, pz = [tuple(pt.ravel().astype(int)) for pt in imgpts]
    cv2.line(img, p0, px, (0,0,255), 2)
    cv2.line(img, p0, py, (0,255,0), 2)
    cv2.line(img, p0, pz, (255,0,0), 2)
```

### 6. Desenha o cubo sobre o marcador

```bash
def draw_cube(img, K, dist, rvec, tvec, side=0.05):
    # Cubo com base no plano do marcador (origem no canto inferior esquerdo do marcador)
    s = side
    # 8 vértices
    objp = np.float32([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],  
        [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]
    ])
    imgpts, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    # base
    cv2.polylines(img, [imgpts[0:4]], True, (255, 255, 255), 2)
    # pilares
    for i in range(4):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i+4]), (200, 200, 200), 2)
    # topo
    cv2.polylines(img, [imgpts[4:8]], True, (160, 160, 160), 2)
```

### 7. Main
```bash
def main():
    ap = argparse.ArgumentParser(description="Realidade Aumentada com ArUco (OpenCV)")
    ap.add_argument("--camera-id", type=int, default=0) #Id da webcam, para definer qual será utilizada
    ap.add_argument("--marker-length", type=float, default=0.02) #Tamanho do marcador
    ap.add_argument("--dict", dest="dict_name", type=str, default="DICT_4X4_50") #Dicionário
    ap.add_argument("--axis-scale", type=float, default=0.03) #Comprimento dos eixos
    ap.add_argument("--cube-size", type=float, default=0.02) #Tamanho do cubo
    args = ap.parse_args()

    # Configura detector
    dictionary = get_aruco_dict(args.dict_name)
    detect = build_detector()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {args.camera_id}")

    # Pegamos um frame inicial para intrínsecos
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Falha ao capturar frame inicial da câmera.")
    
    h, w = frame.shape[:2]
    K, dist = estimate_intrinsics_approx(w, h)
    print("[INFO] Matriz de câmera inicial:\n", K)
    print("[INFO] Coef. de distorção:\n", dist.ravel())

    # Definição dos pontos 3D do marcador (cantos no sistema do marcador)
    marker_len = float(args.marker_length)
    objp = np.array([
        [0, 0, 0],
        [marker_len, 0, 0],
        [marker_len, marker_len, 0],
        [0, marker_len, 0]
    ], dtype=np.float32)

    last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read() #Entrada de dados em tempo real
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Conversão para cinza
        corners, ids, _ = detect(gray, dictionary) # Identifica marcadores e seus vértices

        if ids is not None and len(ids) > 0:
            # desenha contornos
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c in corners:
                c = c.reshape(4, 2).astype(np.float32)
                imgp = c
                success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not success:
                    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
                if success: # Desenha o cubo e os eixos caso o PnP resulte em sucesso
                    draw_axes(frame, K, dist, rvec, tvec, axis_len=args.axis_scale)
                    draw_cube(frame, K, dist, rvec, tvec, side=args.cube_size)

        
        cv2.imshow("AR ArUco - OpenCV", frame) # Atualiza a visualização
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```
# Referências e fontes:

https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/
https://chev.me/arucogen/
https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/
https://en.wikipedia.org/wiki/Perspective-n-Point
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
https://pt.wikipedia.org/wiki/Realidade_aumentada
