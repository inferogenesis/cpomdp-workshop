import re

import pytest

from cpomdp_workshop.symptoms import SYMPTOMS, run, symptoms

RAISING = [s for s in SYMPTOMS if s.wrong is not None]


@pytest.mark.parametrize("symptom", RAISING, ids=lambda s: s.cause)
def test_the_wrong_line_raises_what_the_symptom_says_you_see(symptom):
    assert symptom.quoted is not None
    assert symptom.quoted in symptom.seen
    with pytest.raises(Exception, match=re.escape(symptom.quoted)):
        run(symptom.wrong)


def test_there_are_five():
    assert len(SYMPTOMS) == 5


def test_the_ellipsis_left_in_a_cell_comes_first():
    assert "ellipsis" in SYMPTOMS[0].seen


def test_every_printed_line_fits_a_narrow_cell(capsys):
    symptoms()
    printed = capsys.readouterr().out
    assert max(len(line) for line in printed.splitlines()) <= 80


def test_the_printout_gives_each_symptom_its_cause_and_fix(capsys):
    symptoms()
    printed = capsys.readouterr().out
    for symptom in SYMPTOMS:
        assert symptom.cause in printed
        assert symptom.fix in printed
