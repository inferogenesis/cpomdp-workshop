import pytest

from cpomdp_workshop.symptoms import SYMPTOMS, run, symptoms


@pytest.mark.parametrize("symptom", SYMPTOMS, ids=lambda s: s.name)
def test_the_wrong_line_raises_the_named_error(symptom):
    with pytest.raises(symptom.error):
        run(symptom.wrong)


@pytest.mark.parametrize("symptom", SYMPTOMS, ids=lambda s: s.name)
def test_the_right_line_runs(symptom):
    run(symptom.right)


def test_there_are_five():
    assert len(SYMPTOMS) == 5


def test_the_printout_quotes_cpomdp_word_for_word(capsys):
    symptoms()
    printed = capsys.readouterr().out
    assert (
        "TypeError: observation_noise takes a matrix, got CallableSensor, an "
        "ObservationModel. Pass it as observation_model." in printed
    )
    assert "no longer raises" not in printed
