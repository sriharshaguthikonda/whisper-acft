#!/usr/bin/env python3
"""Regression tests for Kaggle chunked dataset publishing helpers."""

from __future__ import annotations

from tools import kaggle_publish_record_chunks_chunked as pub


def test_validate_kaggle_dataset_id_rejects_long_slug() -> None:
    too_long = "drsriharshaguthik/acft-moonshine-record-test-chunks-manifests-chunked"

    try:
        pub.validate_kaggle_dataset_id(too_long)
    except ValueError as exc:
        assert "between 6 and 50" in str(exc)
    else:
        raise AssertionError("expected long Kaggle dataset slug to fail preflight")


def test_validate_kaggle_dataset_id_rejects_unsafe_slug_chars() -> None:
    try:
        pub.validate_kaggle_dataset_id("drsriharshaguthik/Unsafe_Record_Chunks")
    except ValueError as exc:
        assert "lowercase letters, numbers, and hyphens" in str(exc)
    else:
        raise AssertionError("expected unsafe Kaggle dataset slug to fail preflight")


def test_build_name_preflight_accepts_current_b450_names() -> None:
    chunks = [
        {
            "chunk_index": 1,
            "dataset_id": "drsriharshaguthik/acft-moonshine-record-test-chunks-audio-b450-001",
        }
    ]

    rows = pub.build_name_preflight(
        chunks,
        manifest_dataset_id="drsriharshaguthik/acft-moonshine-record-test-chunks-manifest-b450",
        kaggle_top_dir="Record_test_chunks",
        chunk_file_limit=5000,
    )

    assert [row["slug_len"] for row in rows] == [48, 47]
    assert all(row["status"] == "ok" for row in rows)


def test_build_path_record_supports_working_extract_root() -> None:
    row = {
        "src": r"I:\Record_test_chunks\visit\sample.wav",
        "rel": "visit/sample.wav",
        "size": 123,
        "chunk_index": 1,
        "chunk_slug": "acft-moonshine-record-test-chunks-audio-chunk-001",
        "dataset_id": "drsriharshaguthik/acft-moonshine-record-test-chunks-audio-chunk-001",
    }

    rec = pub.build_kaggle_path_record(
        row,
        kaggle_top_dir="Record_test_chunks",
        audio_path_root_template="/kaggle/working/acft_chunks/{top_dir}",
    )

    assert rec["kaggle_path"] == "/kaggle/working/acft_chunks/Record_test_chunks/visit/sample.wav"
    assert rec["kaggle_uploaded_archive_name"] == "Record_test_chunks.tar"
    assert rec["kaggle_input_extracted_path"] == (
        "/kaggle/input/acft-moonshine-record-test-chunks-audio-chunk-001/visit/sample.wav"
    )
    assert rec["archive_member"] == "visit/sample.wav"
    assert rec["desired_relative_path"] == "Record_test_chunks/visit/sample.wav"


def test_build_path_record_defaults_to_reconstructed_working_paths() -> None:
    row = {
        "src": r"I:\Record_chunks\visit\sample.wav",
        "rel": "visit/sample.wav",
        "size": 123,
        "chunk_index": 1,
        "chunk_slug": "acft-moonshine-record-chunks-audio-chunk-001",
        "dataset_id": "drsriharshaguthik/acft-moonshine-record-chunks-audio-chunk-001",
    }

    rec = pub.build_kaggle_path_record(row, kaggle_top_dir="Record_chunks")

    assert rec["kaggle_path"] == (
        "/kaggle/working/acft_chunks/Record_chunks/visit/sample.wav"
    )
    assert rec["kaggle_uploaded_archive_name"] == "Record_chunks.tar"
    assert rec["kaggle_input_extracted_path"] == (
        "/kaggle/input/acft-moonshine-record-chunks-audio-chunk-001/visit/sample.wav"
    )
    assert rec["archive_member"] == "visit/sample.wav"
    assert rec["desired_relative_path"] == "Record_chunks/visit/sample.wav"


def test_dataset_files_listing_ready_requires_expected_fragment() -> None:
    seed_listing = """name       size  creationDate
---------  ----  --------------------------
_seed.txt   120  2026-06-28 19:27:49.568000
"""
    extracted_listing = """name                size       creationDate
------------------  ---------  --------------------------
visit/sample.wav    123        2026-06-28 19:48:25.053000
"""

    assert not pub.dataset_files_listing_ready(seed_listing, expected_fragment="sample.wav")
    assert pub.dataset_files_listing_ready(extracted_listing, expected_fragment="sample.wav")
    assert pub.dataset_files_listing_ready(extracted_listing, expected_fragment=None)


def test_assign_chunks_honors_byte_limit() -> None:
    entries = [
        {"src": "a.wav", "rel": "a.wav", "size": 70},
        {"src": "b.wav", "rel": "b.wav", "size": 20},
        {"src": "c.wav", "rel": "c.wav", "size": 60},
        {"src": "d.wav", "rel": "d.wav", "size": 40},
    ]

    assigned, summary = pub.assign_chunks(entries, chunk_file_limit=10, chunk_bytes_limit=100)

    assert summary["chunks_total"] == 2
    assert [chunk["bytes"] for chunk in summary["chunks"]] == [90, 100]
    by_rel = {row["rel"]: row["chunk_index"] for row in assigned}
    assert by_rel == {"a.wav": 1, "b.wav": 1, "c.wav": 2, "d.wav": 2}
