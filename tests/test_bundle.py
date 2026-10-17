import json
import stat
import zipfile

import pytest

from ehr_forecast.bundle import unpack


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\b", "a/../b"])
def test_unsafe_archive_paths_rejected_before_extraction(tmp_path, name):
    archive = tmp_path / "model.zip"
    member = zipfile.ZipInfo()
    member.filename = name
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked").exists()


def test_archive_symlink_rejected_before_extraction(tmp_path):
    archive = tmp_path / "model.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, "outside")
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked").exists()


def test_unlisted_archive_member_rejected_before_extraction(tmp_path):
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("manifest.json", json.dumps({"files": {}}))
        stream.writestr("unexpected.csv", "not a model")
    with pytest.raises(ValueError, match="contents"):
        unpack(archive, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked").exists()
