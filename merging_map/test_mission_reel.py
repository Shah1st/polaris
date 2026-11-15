"""Tests of the mission loop for the real rover (audit mm-mission F1, F2, F6; mm-real F6):
an unreachable rover, rover server errors, an operator stop, no progress, and a mission
result that tells success from giving up. Real adapter (RaspRoverReel) against the fake
rover server of test_rover_reel, overhead camera faked, ARENA settings."""

import math

import pytest
from cerveau import ETAT_SUIVI, mission
from lancer_overhead import TOLERANCE_PETITE_ZONE, preparer
from mission_settings import ARENA
from rover_reel import ServeurRoverError
from test_lancer_overhead import FausseCamera
from test_rover_reel import FauxServeur

VICTIM = (2.0, 2.0)


def run(faux, max_tours=600, settings=ARENA, avant=None):
    camera = FausseCamera(faux)
    options = {'horloge': faux.horloge, 'dormir': faux.dormir}
    carte, rover = preparer(camera, faux.requete, options_rover=options)
    if avant:
        avant(rover)
    lines = []
    res = mission(
        rover,
        carte,
        (0.4, 0.4),
        VICTIM,
        zone=camera.zone(),
        tolerance=TOLERANCE_PETITE_ZONE,
        journal=lines.append,
        settings=settings,
        max_tours=max_tours,
    )
    return res, '\n'.join(lines)


def test_a_normal_mission_reports_its_success() -> None:
    res, _ = run(FauxServeur(0.4, 0.4, math.pi))
    assert res['issue'] == 'arrivee' and res['succes'] and ETAT_SUIVI[res['issue']] == 'arrive'


def test_a_rover_lost_for_good_ends_the_mission() -> None:
    """The audit's bench ran 600 turns (10.3 h simulated) and logged 'liaison retrouvee'
    600 times; now the mission stops after the adapter's bounded wait."""
    faux = FauxServeur(0.4, 0.4, math.pi)
    faux.coupure = (8.0, math.inf)
    res, log = run(faux)
    assert res['issue'] == 'rover_injoignable' and not res['succes']
    assert ETAT_SUIVI[res['issue']] == 'abandon'
    assert 'abandon: rover injoignable' in log
    assert 'liaison rover retrouvee' not in log
    assert log.count('LIAISON ROVER PERDUE') == 1
    assert faux.t < 120.0  # 60 s of waiting for the link, not hours


def test_a_short_cut_still_comes_back() -> None:
    faux = FauxServeur(0.4, 0.4, math.pi)
    faux.coupure = (8.0, 30.0)
    res, log = run(faux)
    assert 'liaison rover retrouvee' in log
    assert res['issue'] == 'arrivee'


def failing_steps(n, erreur):
    """Makes the n first robot.deplacer() calls raise `erreur`."""
    compte = {'appels': 0}

    def installer(rover):
        vrai = rover.deplacer

        def deplacer(cible):
            compte['appels'] += 1
            if compte['appels'] <= n:
                raise erreur
            return vrai(cible)

        rover.deplacer = deplacer

    return installer, compte


def test_a_rover_server_error_is_retried() -> None:
    installer, _ = failing_steps(2, ServeurRoverError('Cap du rover impossible a mesurer'))
    faux = FauxServeur(0.4, 0.4, math.pi)
    res, log = run(faux, avant=installer)
    assert log.count('ERREUR du rover') == 2 and 'Cap du rover impossible' in log
    assert res['issue'] == 'arrivee'
    assert any(r[2] == '/stop' for r in faux.requetes)  # stopped before trying again


def test_repeated_rover_server_errors_end_the_mission() -> None:
    installer, compte = failing_steps(100, ServeurRoverError('HTTP 500'))
    res, log = run(FauxServeur(0.4, 0.4, math.pi), avant=installer)
    assert res['issue'] == 'erreur_robot' and not res['succes']
    assert ETAT_SUIVI[res['issue']] == 'erreur'
    assert compte['appels'] == ARENA.robot_error_retries + 1


class ArretOperateur(Exception):  # noqa: N818 - interface 5, matched by name
    pass


class ArretOperateurDerive(ServeurRoverError):  # noqa: N818 - even if it ever derives from it
    pass


ArretOperateurDerive.__name__ = 'ArretOperateur'


@pytest.mark.parametrize('classe', [ArretOperateur, ArretOperateurDerive])
def test_an_operator_stop_propagates_at_once(classe) -> None:
    installer, compte = failing_steps(100, classe('stopped'))
    with pytest.raises(classe):
        run(FauxServeur(0.4, 0.4, math.pi), avant=installer)
    assert compte['appels'] == 1


class RouesBloquees(FauxServeur):
    """The wheels stop turning at t_bloque (motor board stuck, flat battery): every step
    ends 'stalled' (from the audit's verify script stall_repete.py)."""

    t_bloque = 8.0

    def _rouler(self):
        if self.t < self.t_bloque:
            return super()._rouler()
        self.tache['immobile'] += self.TICK
        if self.tache['immobile'] > self.STALL:
            self._finir('stalled')
        return None


def test_stuck_wheels_end_with_no_progress() -> None:
    """The audit's bench ran 600 turns of 'choc' (26 min simulated)."""
    faux = RouesBloquees(0.4, 0.4, math.pi)
    res, log = run(faux)
    assert res['issue'] == 'sans_progres' and not res['succes']
    assert "abandon: le rover n'avance plus" in log
    assert log.count('CHOC') <= ARENA.same_bump_max + ARENA.stall_steps
    assert faux.t < 200.0


def test_max_turns_is_not_a_success() -> None:
    # a wall the overhead camera does not see: the first turn ends with a bump
    res, log = run(FauxServeur(0.4, 0.4, math.pi, mur_x=1.0), max_tours=1)
    assert 'CHOC' in log
    assert res['issue'] == 'max_tours' and not res['succes']
    assert 'abandon: trop de tours' in log
