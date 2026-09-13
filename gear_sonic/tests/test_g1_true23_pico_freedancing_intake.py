import zipfile

import pytest

from gear_sonic.scripts.fetch_g1_true23_pico_freedancing_sample import PAIR_NAMES, select_pair


def member(name, size=100):
    result = zipfile.ZipInfo(name)
    result.file_size, result.compress_size = size, 50
    return result


def test_first_complete_pair_selected_before_content():
    entries = [member(f"subject02/motion03/{name}") for name in PAIR_NAMES]
    entries += [member(f"subject01/motion02/{name}") for name in PAIR_NAMES]
    entries += [member("subject01/motion01/gt_body_parms.pt")]
    path, selected, count = select_pair(entries)
    assert path == "subject01/motion02" and count == 2
    assert [x.filename for x in selected] == [f"{path}/{name}" for name in PAIR_NAMES]


@pytest.mark.parametrize("name", ["../outside", "/absolute", "folder\\escape"])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError, match="unsafe"):
        select_pair([member(name)])


def test_duplicate_paths_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select_pair([member("same"), member("same")])


@pytest.mark.parametrize("size", [0, 64 * 1024 * 1024 + 1, 60000])
def test_empty_oversized_or_bomb_rejected(size):
    entries = [member(name, size) for name in PAIR_NAMES]
    with pytest.raises(ValueError, match="bounded"):
        select_pair(entries)
