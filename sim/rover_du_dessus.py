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
Measured on the five real pictures from above we have: the rover found in all five (scores
0.32 to 0.59), nothing found in four pictures without it, 0.7 s per picture on an Apple M5.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

MODELE = 'google/owlv2-base-patch16-ensemble'
# Measured on 5 real pictures with the rover and 4 without: "a robot" scores 0.32 to 0.59 on
# the rover and 0.16 at most anywhere else; the longer descriptions tried first ("a small black
# robot car") dropped to 0.03 on the rover seen from further away.
ROVER = ['a robot', 'a small black robot', 'a remote controlled car']
AUTRES = ['a shoe', 'a cardboard box', 'a bag', 'a person', 'a backpack']


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


SOMBRE_MIN = 0.25  # share of dark pixels in the rover's box (measured: 0.45 to 0.7 on it)


def part_sombre(image, boite):
    """Share of the box's pixels that are dark (the rover's black body)."""
    x0, y0, x1, y1 = (int(round(v)) for v in boite)
    morceau = image[max(y0, 0) : y1, max(x0, 0) : x1]
    if morceau.size == 0:
        return 0.0
    return float((cv2.cvtColor(morceau, cv2.COLOR_BGR2GRAY) < 70).mean())


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class DetecteurRover:
    def __init__(self, seuil=0.2, appareil=None, marqueur_devant=False):
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
        # every box with its score for every name (the usual post-processing keeps only the
        # best name of a box: the rover under its cables is as much "a bag" as "a robot")
        scores = self.torch.sigmoid(sortie.logits[0]).cpu().numpy()
        cote = max(h, w)  # the picture is padded to a square; boxes are fractions of it
        cx, cy, bw, bh = (sortie.pred_boxes[0].cpu().numpy() * cote).T
        n = len(ROVER)
        trouves = [
            (float(scores[i, :n].max()), float(scores[i, n:].max()),
             (cx[i] - bw[i] / 2, cy[i] - bh[i] / 2, cx[i] + bw[i] / 2, cy[i] + bh[i] / 2))
            for i in np.nonzero(scores[:, :n].max(axis=1) >= 0.1)[0]
        ]  # fmt: skip
        return choisir(image, trouves, self.seuil, self.marqueur_devant)


def choisir(image, trouves, seuil=0.2, marqueur_devant=False):
    """The rover among the model's boxes (score as a robot, best score as something else,
    box): the best "robot" box above `seuil`, that is not clearly something else (a shoe, a
    bag...) and not huge."""
    h, w = image.shape[:2]
    for score, autre, boite in sorted(trouves, reverse=True):
        if score < seuil or score < 0.8 * autre:
            continue
        bw, bh = boite[2] - boite[0], boite[3] - boite[1]
        if bw <= 4 or bh <= 4 or bw * bh > 0.15 * w * h or max(bw, bh) > 3 * min(bw, bh):
            continue
        if part_sombre(image, boite) < SOMBRE_MIN:
            continue  # the rover is black: a robot drawn on a banner or a poster is not
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
