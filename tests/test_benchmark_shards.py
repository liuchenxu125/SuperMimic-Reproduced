import json

import pytest

from supermimic.eval.benchmark import clip_result, save_results
from supermimic.eval.shards import hashlib_names, merge_shards


def shards(tmp_path):
    paths = [tmp_path / f"{index}.json" for index in range(2)]
    rows = [clip_result("a", 100, 10, 0.8, True), clip_result("b", 50, 50, 2.5, False)]
    for index, (path, row) in enumerate(zip(paths, rows)):
        save_results(
            [row],
            path,
            meta={"sha256": "same-weights"},
            evaluation={"robot": "g1"},
            shard={
                "index": index,
                "count": 2,
                "total_clips": 2,
                "dataset_sha256": hashlib_names(["a", "b"]),
            },
        )
    return paths


def test_merge_preserves_full_clip_scoring(tmp_path):
    result = merge_shards(shards(tmp_path))
    assert result["summary"]["failed"] == 1
    assert result["summary"]["mean_norm_return"] == pytest.approx(0.29)
    assert result["summary"]["frame_weighted_norm_return"] == pytest.approx(0.22)
    assert result["merge"]["indexed_completeness_checked"]


@pytest.mark.parametrize(
    "fault", ["duplicate", "checkpoint", "protocol", "missing", "score", "dataset"]
)
def test_merge_rejects_invalid_or_mixed_reports(tmp_path, fault):
    paths = shards(tmp_path)
    d = json.loads(paths[1].read_text())
    if fault == "duplicate":
        d["clips"][0]["name"] = "a"
    if fault == "checkpoint":
        d["checkpoint"]["sha256"] = "different-weights"
    if fault == "protocol":
        d["evaluation"]["robot"] = "casbot02"
    if fault == "score":
        d["clips"][0]["norm_return"] = 9.0
    if fault == "dataset":
        d["clips"][0]["name"] = "c"
    paths[1].write_text(json.dumps(d))
    if fault == "missing":
        paths = paths[:1]
    with pytest.raises(ValueError):
        merge_shards(paths)


def test_legacy_requires_explicit_conversion(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "name": "failed",
                        "frames": 100,
                        "steps": 10,
                        "norm_return": 0.8,
                        "failed": True,
                        "fail_step": 10,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="legacy-survived"):
        merge_shards([path])
    result = merge_shards([path], legacy_survived=True)
    assert result["clips"][0]["norm_return"] == pytest.approx(0.08)
    assert result["clips"][0]["norm_return_survived"] == pytest.approx(0.8)


def test_eval_cli_assigns_disjoint_shards_and_records_checkpoint_bytes(tmp_path, monkeypatch):
    from scripts import eval_benchmark as cli

    data = tmp_path / "data"
    data.mkdir()
    for name in ["c", "a", "b"]:
        (data / f"{name}.npz").touch()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed weights")
    def load_open_checkpoint(stream, device):
        assert stream.read() == b"fixed weights"
        return None, {"env": {"future_frames": 10}, "_checkpoint": {"iteration": 42}}

    monkeypatch.setattr(
        cli,
        "load_policy_from_checkpoint",
        load_open_checkpoint,
    )
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda policy, files, **kwargs: [
            clip_result(cli.Path(path).stem, 2, 2, 0.1, False) for path in files
        ],
    )
    paths = [tmp_path / f"shard_{index}.json" for index in range(2)]
    for index, path in enumerate(paths):
        cli.main(checkpoint, npz_dir=data, num_shards=2, shard_index=index, out=path)
    result = merge_shards(paths)
    assert [row["name"] for row in result["clips"]] == ["a", "b", "c"]
    assert result["checkpoint"]["sha256"] == cli.hashlib.sha256(b"fixed weights").hexdigest()
    assert result["evaluation"]["robot"] == "g1"
