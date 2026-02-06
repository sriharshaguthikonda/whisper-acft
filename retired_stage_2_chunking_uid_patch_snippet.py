"""Drop-in snippet for Stage 2 (chunking) right when you create each chunk row.

You already have (or can compute):
  - orig_audio_path
  - chunk index
  - core_start/core_end

Add:
  - base_uid (stable forever)
  - uid (== base_uid for originals)
  - aug_stage (empty for originals)
  - parent_uid (empty for originals)

Also: include base_uid in the output chunk filename so downstream never loses it.
"""

# Example inside your loop that produces chunk rows:
#
# base_uid = make_base_uid(orig_audio_path, chunk_index, core_start, core_end)
# out_wav = f"{stem}__chunk{chunk_index:04d}__uid{base_uid}.wav"
# row = {
#   "orig_audio_path": orig_audio_path,
#   "chunk_index": chunk_index,
#   "core_start": core_start,
#   "core_end": core_end,
#   "out_wav": out_wav,
#   "base_uid": base_uid,
#   "uid": base_uid,
#   "aug_stage": "",
#   "aug_copy_idx": 0,
#   "parent_uid": "",
# }
