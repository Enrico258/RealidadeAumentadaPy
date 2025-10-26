from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

#Aruco
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


#Define o detector aruco
def build_detector(params: dict | None = None):
    aruco_params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters') else cv2.aruco.DetectorParameters_create()
    if params:
        for k, v in params.items():
            if hasattr(aruco_params, k):
                setattr(aruco_params, k, v)
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


#Estiima uma calibração para a webcam
def estimate_intrinsics_approx(w: int, h: int, fov_deg: float = 60.0):
    f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((1, 5), dtype=np.float32)
    return K, dist

def draw_axes(img, K, dist, rvec, tvec, axis_len=0.03):
    # Eixos X (vermelho), Y (verde), Z (azul) — cores padrão do OpenCV
    axis = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    p0, px, py, pz = [tuple(pt.ravel().astype(int)) for pt in imgpts]
    cv2.line(img, p0, px, (0,0,255), 2)
    cv2.line(img, p0, py, (0,255,0), 2)
    cv2.line(img, p0, pz, (255,0,0), 2)


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
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray, dictionary)

        if ids is not None and len(ids) > 0:
            # desenha contornos
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c in corners:
                c = c.reshape(4, 2).astype(np.float32)
                # solvePnP precisa de correspondência 2D-3D coerente
                # Convenção: canto 0 -> (0,0), 1 -> (L,0), 2 -> (L,L), 3 -> (0,L)
                imgp = c
                success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not success:
                    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
                if success:
                    draw_axes(frame, K, dist, rvec, tvec, axis_len=args.axis_scale)
                    draw_cube(frame, K, dist, rvec, tvec, side=args.cube_size)

        
        cv2.imshow("AR ArUco - OpenCV", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

