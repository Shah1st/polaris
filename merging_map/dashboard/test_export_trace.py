"""Chaque evenement que la mission peut emettre a un titre et un texte pour la page."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_trace  # noqa: E402
from cerveau import RESUMES  # noqa: E402
from fusion_carte import CarteFusion  # noqa: E402


class SimuFactice:
    """Juste ce que textes() lit sur le rover simule."""

    champ, sigma = 90, 0.12

    def __init__(self, connecte):
        self._connecte = connecte

    def connecte(self):
        return self._connecte


# Les infos que la mission passe avec chaque type d'evenement (cerveau.mission).
INFOS = {
    'zoom': {'tuiles': [(2, 0, 0)]},
    'suivre': {'chemin': [(0.0, 0.0), (1.0, 0.0)], 'premier': True},
    'degrade': {'restant': 1.5},
    'fin': {'restant': 1.5},
    'arrivee': {'distance': 0.3},
    'retour': {'sigma': 0.6},
    'poursuite': {'sigma': 0.6},
    'reconnexion': {'erreur': 0.4},
    'perte_rover': {'instant': 12.0},
    'reprise_rover': {'zone': (1.0, 2.0, 0.5)},
}


@pytest.mark.parametrize('connecte', [True, False])
@pytest.mark.parametrize('type_', sorted(RESUMES))
def test_chaque_type_de_la_mission_a_un_texte(type_, connecte):
    infos = {'message': 'raison du test', **INFOS.get(type_, {})}
    titre, texte = export_trace.textes(type_, SimuFactice(connecte), CarteFusion(), infos, 2.0)
    assert titre and isinstance(titre, str)
    assert isinstance(texte, str)


def test_type_inconnu_refuse():
    with pytest.raises(ValueError):
        export_trace.textes('pas_un_type', SimuFactice(True), CarteFusion(), {}, 0.0)
