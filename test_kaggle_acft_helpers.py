from __future__ import annotations

import json
import re
from pathlib import Path

from tools import kaggle_acft_helpers as kh


def test_parse_canonical_dataset_handles_stops_before_non_canonical(tmp_path: Path) -> None:
    doc = tmp_path / "export.md"
    doc.write_text(
        "\n".join(
            [
                "# export",
                "## Canonical Kaggle Datasets",
                "- `drsriharshaguthik/acft-moonshine-primary-refs-noise`",
                "text",
                "- `drsriharshaguthik/acft-moonshine-src-record-harsha-001`",
                "## Non-Canonical Kaggle Attempts",
                "- `drsriharshaguthik/old-attempt-do-not-use`",
            ]
        ),
        encoding="utf-8",
    )

    assert kh.parse_canonical_dataset_handles(doc) == [
        "drsriharshaguthik/acft-moonshine-primary-refs-noise",
        "drsriharshaguthik/acft-moonshine-src-record-harsha-001",
    ]


def test_reconstruct_kaggle_sources_preserves_top_level_paths(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    working = tmp_path / "working"
    (input_root / "acft-moonshine-src-record-harsha-001" / "Record_harsha").mkdir(parents=True)
    (input_root / "acft-moonshine-src-record-harsha-001" / "Record_harsha" / "a.wav").write_bytes(b"audio")
    (input_root / "acft-moonshine-primary-refs-noise" / "noise" / "RIRS_NOISES").mkdir(parents=True)
    (input_root / "acft-moonshine-primary-refs-noise" / "noise" / "RIRS_NOISES" / "r.wav").write_bytes(b"rir")
    (input_root / "acft-moonshine-primary-refs-noise" / "whisper-acft").mkdir(parents=True)
    (input_root / "acft-moonshine-primary-refs-noise" / "whisper-acft" / "speaker_sort_scores.csv").write_text(
        "speaker,score\n", encoding="utf-8"
    )

    result = kh.reconstruct_kaggle_sources(
        input_root=input_root,
        output_root=working,
        dataset_handles=[
            "drsriharshaguthik/acft-moonshine-src-record-harsha-001",
            "drsriharshaguthik/acft-moonshine-primary-refs-noise",
        ],
    )

    assert result.copied_files == 3
    assert (working / "Record_harsha" / "a.wav").read_bytes() == b"audio"
    assert (working / "noise" / "RIRS_NOISES" / "r.wav").read_bytes() == b"rir"
    assert (working / "whisper-acft" / "speaker_sort_scores.csv").exists()

    inventory = [json.loads(line) for line in (working / "_kaggle_source_inventory.jsonl").read_text().splitlines()]
    assert {row["top_level"] for row in inventory} == {"Record_harsha", "noise", "whisper-acft"}

    second = kh.reconstruct_kaggle_sources(
        input_root=input_root,
        output_root=working,
        dataset_handles=[
            "drsriharshaguthik/acft-moonshine-src-record-harsha-001",
            "drsriharshaguthik/acft-moonshine-primary-refs-noise",
        ],
    )
    assert second.copied_files == 0
    assert second.skipped_existing_files == 3


def test_stage_signature_and_resume_gate(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "out.jsonl"
    state_dir = tmp_path / "state"
    inp.write_text('{"x":1}\n', encoding="utf-8")
    out.write_text("done\n", encoding="utf-8")

    sig = kh.stage_signature(inputs=[inp], config={"profile": "smoke", "limit": 5})
    assert kh.should_run_stage("stage1", outputs=[out], signature=sig, state_dir=state_dir)

    kh.mark_stage_done("stage1", outputs=[out], signature=sig, state_dir=state_dir)
    assert not kh.should_run_stage("stage1", outputs=[out], signature=sig, state_dir=state_dir)

    changed = kh.stage_signature(inputs=[inp], config={"profile": "smoke", "limit": 6})
    assert kh.should_run_stage("stage1", outputs=[out], signature=changed, state_dir=state_dir)


def test_dataset_handle_and_dry_run_metadata(tmp_path: Path) -> None:
    handle = kh.make_dataset_handle(
        owner="drsriharshaguthik",
        prefix="acft-kaggle-chunks",
        run_tag="20260630_very_long_run_tag_with_extra_words",
        suffix="stage-01-source-reconstruct",
    )

    owner, slug = handle.split("/", 1)
    assert owner == "drsriharshaguthik"
    assert len(slug) <= kh.KAGGLE_DATASET_SLUG_MAX
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*[a-z0-9]", slug)

    local_dir = tmp_path / "dataset"
    local_dir.mkdir()
    (local_dir / "manifest.jsonl").write_text("{}\n", encoding="utf-8")

    plan = kh.write_dataset_metadata(
        local_dir=local_dir,
        handle=handle,
        title="ACFT Kaggle chunks smoke",
        subtitle="dry run",
        keywords=["acft", "whisper"],
        licenses=[{"name": "unknown"}],
    )

    assert plan["metadata_path"] == str(local_dir / "dataset-metadata.json")
    metadata = json.loads((local_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == handle
    assert metadata["isPrivate"] is True

    publish = kh.publish_dataset(local_dir=local_dir, handle=handle, version_notes="stage done", dry_run=True)
    assert publish["dry_run"] is True
    assert publish["handle"] == handle
    assert (local_dir / "_publish_plan.json").exists()


def test_cap_public_mix_respects_ratio_and_tags_rows() -> None:
    private_rows = [{"id": f"p{i}", "audio_path": f"private/{i}.wav"} for i in range(10)]
    public_rows = [{"id": f"u{i}", "audio_path": f"public/{i}.wav"} for i in range(100)]

    first = kh.mix_private_and_public_rows(private_rows, public_rows, public_ratio=0.30, seed=17)
    second = kh.mix_private_and_public_rows(private_rows, public_rows, public_ratio=0.30, seed=17)

    public_count = sum(1 for row in first if row["dataset_scope"] == "public_asr")
    private_count = sum(1 for row in first if row["dataset_scope"] == "private_acft")
    assert private_count == 10
    assert public_count == 4
    assert public_count / len(first) <= 0.30
    assert first == second
    assert all(row.get("exclude_from_private_eval") is True for row in first if row["dataset_scope"] == "public_asr")


def test_write_stage12_smoke_fixture_creates_stage_inputs(tmp_path: Path) -> None:
    fixture = kh.write_stage12_smoke_fixture(tmp_path / "fixture")

    audio = Path(fixture["audio_path"])
    transcript = Path(fixture["transcript_path"])
    assert audio.exists()
    assert transcript.exists()
    assert Path(fixture["chunks_dir"]).is_dir()

    data = json.loads(transcript.read_text(encoding="utf-8"))
    assert data["input_file"]["path"] == str(audio)
    assert data["groq_response"]["segments"][0]["text"]
