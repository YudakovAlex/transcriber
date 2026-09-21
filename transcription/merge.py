"""The original overlap rule, kept identical for local/remote comparisons."""

from .contracts import validate_result


def merge_chunk_results(chunk_results, overlap_sec):
    if not chunk_results:
        return ""
    if len(chunk_results) == 1:
        return (chunk_results[0].get("text") or "").strip()
    parts = []
    for i, result in enumerate(chunk_results):
        text = result.get("text") or ""
        segments = result.get("segments") or []
        if text.strip() and not segments:
            raise ValueError("Nonempty multi-chunk text has no segments; refusing to lose text.")
        if i == 0:
            parts.append(text.strip())
        else:
            parts.append(" ".join((s.get("text") or "").strip() for s in segments
                                  if s.get("end", 0) > overlap_sec).strip())
    return " ".join(p for p in parts if p).strip()


def ordered_results(requests, results):
    indexed = {}
    expected = {r["chunk_index"]: r for r in requests}
    for result in results:
        index = result.get("chunk_index")
        if index not in expected:
            raise ValueError("Unexpected chunk index.")
        validate_result(result, expected[index])
        if index in indexed and indexed[index] != result:
            raise ValueError("Conflicting duplicate chunk results.")
        indexed[index] = result
    if set(indexed) != set(expected):
        raise ValueError("Cannot merge an incomplete recording.")
    return [indexed[i] for i in sorted(indexed)]
