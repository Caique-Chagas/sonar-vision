import pytest

from sonar_vision.br.lineage import validate_lineage


def record(source="1", group="author|2020-01-01", split="train"):
    return {"id": source, "group_id": group, "split": split}


def test_same_training_role_can_be_reused():
    assert validate_lineage([record()], [record()])["images"] == 1


@pytest.mark.parametrize("key", ["id", "group_id"])
def test_historical_test_cannot_leak_through_resizing(key):
    old = record(split="test")
    new = record(source="other", group="otherday")
    new[key] = old[key]
    with pytest.raises(ValueError, match="reservation"):
        validate_lineage([new], [old])


@pytest.mark.parametrize("old_split", ["train", "val", "test", "reserved"])
def test_historical_source_cannot_be_called_new_test(old_split):
    with pytest.raises(ValueError, match="independent test"):
        validate_lineage([record(split="test")], [record(split=old_split)])


def test_validation_capture_cannot_move_to_train():
    with pytest.raises(ValueError, match="role conflict"):
        validate_lineage([record(source="new")], [record(split="val")])


def test_capture_split_crossing_rejected():
    with pytest.raises(ValueError, match="crosses splits"):
        validate_lineage([record(), record(source="2", split="val")], [])


def test_resized_source_duplicates_rejected():
    with pytest.raises(ValueError, match="Repeated source"):
        validate_lineage([record(), record(group="other")], [])


@pytest.mark.parametrize("field", ["id", "group_id", "split"])
def test_missing_lineage_rejected(field):
    row = record()
    row[field] = ""
    with pytest.raises(ValueError, match="required"):
        validate_lineage([row], [])
