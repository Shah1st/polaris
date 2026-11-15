"""
rover_du_dessus.py - Find the rover in a picture taken from above, with a vision model, when
its ArUco marker is too small to be read (a phone a few metres up: the marker is 15 px wide).

    detecteur = DetecteurRover()                 # loads OWLv2 once (Apple GPU, CUDA or CPU)
    vu = detecteur(image_bgr)                    # None, or a RoverVu

- where: OWLv2 (open vocabulary detector, google/owlv2-base-patch16-ensemble) asked for a small
  robot car. It is also asked for shoes, bags and boxes, so that those are named as such
  instead of being taken for the rover;
- heading: the white paper of the marker inside the rover's box. The marker sits at the back
  of the rover: the direction from the paper to the centre of the box is the heading. It is
  approximate (10 to 20 degrees), and None when no white patch is found;
- scale: the box is about the size of the rover, which gives the metres per pixel.

Needs `transformers` (pip install transformers) and about 620 MB of weights at the first run.
Measured on the two real pictures from above we have: the rover found in both (scores 0.44
and 0.36), 0.7 s per picture on an Apple M5.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

MODELE = 'google/owlv2-base-patch16-ensemble'
ROVER = ['a small black robot car', 'a toy tank robot']
AUTRES = ['a shoe', 'a cardboard box', 'a bag', 'a person']


@dataclass
class RoverVu:
    x: float  # centre of the box, pixels
    y: float
    cap: float | None  # heading in the picture (radians, x right, y DOWN); None: unknown
    taille: float  # side of the box, pixels: sqrt(width * height)
    score: float
    boite: tuple[float, float, float, float]


def cap_par_papier(image, boite):
    """Direction from the marker's white paper to the centre of the rover's box, or None."""
    x0, y0, x1, y1 = (int(round(v)) for v in boite)
    x0, y0 = max(x0, 0), max(y0, 0)
    morceau = image[y0:y1, x0:x1]
    if morceau.size == 0:
        return None
    hsv = cv2.cvtColor(morceau, cv2.COLOR_BGR2HSV)
    blanc = ((hsv[..., 2] > 165) & (hsv[..., 1] < 80)).astype(np.uint8)
    n, _, stats, centres = cv2.connectedComponentsWithStats(blanc)
    surface = morceau.shape[0] * morceau.shape[1]
    choix = None
    for i in range(1, n):
        aire = stats[i, cv2.CC_STAT_AREA]
        if 0.015 * surface < aire < 0.35 * surface and (choix is None or aire > choix[0]):
            choix = (aire, centres[i])
    if choix is None:
        return None
    cx, cy = morceau.shape[1] / 2, morceau.shape[0] / 2
    dx, dy = cx - choix[1][0], cy - choix[1][1]
    if math.hypot(dx, dy) < 0.08 * math.sqrt(surface):
        return None  # the paper is at the centre: no direction to read
    return math.atan2(dy, dx)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class DetecteurRover:
    def __init__(self, seuil=0.25, appareil=None, marqueur_devant=False):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        if appareil is None:
            appareil = (
                'cuda'
                if torch.cuda.is_available()
                else 'mps'
                if torch.backends.mps.is_available()
                else 'cpu'
            )
        self.appareil = appareil
        self.seuil = seuil
        self.marqueur_devant = marqueur_devant
        self.proc = Owlv2Processor.from_pretrained(MODELE)
        self.modele = Owlv2ForObjectDetection.from_pretrained(MODELE).to(appareil).eval()

    def __call__(self, image):
        from PIL import Image

        h, w = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        entree = self.proc(text=[ROVER + AUTRES], images=pil, return_tensors='pt').to(
            self.appareil
        )
        with self.torch.no_grad():
            sortie = self.modele(**entree)
        res = self.proc.post_process_grounded_object_detection(
            sortie, threshold=0.1, target_sizes=[(h, w)]
        )[0]
        trouves = list(
            zip(res['scores'].tolist(), res['labels'].tolist(), res['boxes'].tolist(), strict=True)
        )
        return choisir(image, trouves, self.seuil, self.marqueur_devant)


def choisir(image, trouves, seuil=0.25, marqueur_devant=False):
    """The rover among the model's boxes (score, label index, box): the best "robot" box above
    `seuil` that no other name claims with a better score, and that is not huge."""
    h, w = image.shape[:2]
    autres = [(s, b) for s, e, b in trouves if e >= len(ROVER)]
    for score, etiquette, boite in sorted(trouves, reverse=True):
        if etiquette >= len(ROVER) or score < seuil:
            continue
        bw, bh = boite[2] - boite[0], boite[3] - boite[1]
        if bw <= 4 or bh <= 4 or bw * bh > 0.15 * w * h or max(bw, bh) > 3 * min(bw, bh):
            continue
        if any(s > score and iou(boite, b) > 0.5 for s, b in autres):
            continue
        cap = cap_par_papier(image, boite)
        if cap is not None and marqueur_devant:
            cap += math.pi
        return RoverVu(
            (boite[0] + boite[2]) / 2,
            (boite[1] + boite[3]) / 2,
            cap,
            math.sqrt(bw * bh),
            score,
            tuple(boite),
        )
    return None
