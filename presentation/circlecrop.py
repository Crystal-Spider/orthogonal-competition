#!/usr/bin/env -S uv run --script --managed-python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "certifi",
#     "numpy",
#     "opencv-python-headless",
# ]
# ///
"""Crop portraits to a circle centred on the face.

Every picture is cropped with the same proportions: the circle diameter is a
fixed multiple of the detected face width, so each face fills the same fraction
of its circle.  Faces are found with OpenCV's YuNet detector, whose model is
downloaded on first run and cached.

    ./circlecrop.py [imgs]
"""

import os
import sys

# --managed-python in the shebang above keeps uv off the system interpreter:
# the nix one cannot load the manylinux wheels of numpy and opencv.
# For the same reason a PYTHONPATH inherited from the shell (nix-shell,
# direnv, ...) has to go, as it shadows the environment uv just built for us.
if os.environ.pop("PYTHONPATH", None):
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])

import argparse
import pathlib
import ssl
import urllib.request

import certifi
import cv2
import numpy as np

MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")


def model_path():
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"))
    path = cache / "circlecrop" / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading the face detection model to {path}", file=sys.stderr)
        tmp = path.with_suffix(".part")
        # our own CA bundle: the interpreter uv downloads has no system one
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(MODEL_URL, context=ctx) as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        tmp.rename(path)
    return str(path)


def detect(model, img):
    """Return the biggest face in the image as YuNet's [x, y, w, h, ...] row."""
    h, w = img.shape[:2]
    # detect on a downscaled copy for speed, then scale the coordinates back
    scale = min(1.0, 640.0 / max(h, w))
    small = img if scale == 1.0 else cv2.resize(img, (round(w * scale), round(h * scale)))
    sh, sw = small.shape[:2]
    det = cv2.FaceDetectorYN.create(model, "", (sw, sh), 0.6, 0.3, 5000)
    _, faces = det.detect(small)
    if faces is None or len(faces) == 0:
        det.setScoreThreshold(0.3)  # second chance on hard pictures
        _, faces = det.detect(small)
    if faces is None or len(faces) == 0:
        return None
    faces = np.array(faces, dtype=float) / scale
    faces[:, 14] *= scale  # undo the scaling of the score column
    return faces[np.argmax(faces[:, 2] * faces[:, 3])]


def crop(img, face, size, radius, offset):
    x, y, w, h = face[:4]
    r = radius * w
    side = 2 * r
    x0, y0 = x + w / 2.0 - r, y + h / 2.0 + offset * h - r
    # keep the window inside the picture when there is room, so that only
    # pictures smaller than the window fall back to border replication
    H, W = img.shape[:2]
    if side <= W:
        x0 = min(max(x0, 0.0), W - side)
    if side <= H:
        y0 = min(max(y0, 0.0), H - side)
    padded = x0 < -0.5 or y0 < -0.5 or x0 + side > W + 0.5 or y0 + side > H + 0.5

    k = size / side
    M = np.array([[k, 0, -x0 * k], [0, k, -y0 * k]], dtype=np.float64)
    square = cv2.warpAffine(img, M, (size, size),
                            flags=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    ss = 4  # supersample the mask to get an antialiased edge
    mask = np.zeros((size * ss, size * ss), np.uint8)
    cv2.circle(mask, (size * ss // 2, size * ss // 2), size * ss // 2 - ss // 2, 255, -1)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    return np.dstack([square, mask]), padded


def main():
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imgs", nargs="?", type=pathlib.Path, default=here / "imgs",
                    help="directory holding the portraits (default: %(default)s)")
    ap.add_argument("--size", type=int, default=640, help="output size in pixels")
    ap.add_argument("--radius", type=float, default=1.03,
                    help="circle radius as a multiple of the face width")
    ap.add_argument("--offset", type=float, default=-0.09,
                    help="circle centre offset from the face centre, in face heights "
                         "(negative moves it up, leaving room for the hair)")
    ap.add_argument("--suffix", default="-circle", help="suffix of the cropped files")
    args = ap.parse_args()

    sources = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                     for p in args.imgs.glob(ext) if not p.stem.endswith(args.suffix))
    if not sources:
        sys.exit(f"no pictures in {args.imgs}")

    model = model_path()
    failed = []
    for path in sources:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            failed.append(path.name)
            print(f"{path.name:35s} unreadable")
            continue
        face = detect(model, img)
        if face is None:
            failed.append(path.name)
            print(f"{path.name:35s} NO FACE FOUND")
            continue
        out, padded = crop(img, face, args.size, args.radius, args.offset)
        # the circle needs transparency outside it, hence png
        dest = path.with_name(path.stem + args.suffix + ".png")
        cv2.imwrite(str(dest), out)
        print(f"{path.name:35s} -> {dest.name}  score={face[14]:.2f}"
              f"{'  (border replicated)' if padded else ''}")

    print(f"\n{len(sources) - len(failed)}/{len(sources)} pictures cropped")
    if failed:
        print("no face found in: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
