from tinylm.tokenizer import (
    UNICODE_PROBES,
    encode_batch,
    load_tokenizer,
    tokenizer_diagnostics,
    tokenizer_fingerprint,
)


def test_special_tokens_have_fixed_ids(tokenizer):
    assert tokenizer.eos_token == "<|endoftext|>" and tokenizer.eos_token_id == 0
    assert tokenizer.pad_token == "<|pad|>" and tokenizer.pad_token_id == 1
    assert tokenizer.bos_token_id == tokenizer.eos_token_id  # one boundary token
    assert tokenizer.unk_token is None  # byte-level: nothing is unknown


def test_encode_adds_no_special_tokens(tokenizer):
    ids = encode_batch(tokenizer, ["Once upon a time"])[0]
    assert tokenizer.eos_token_id not in ids and tokenizer.pad_token_id not in ids


def test_roundtrip_is_exact_including_unseen_unicode(tokenizer):
    # The fixture corpus is ASCII-only; byte-level BPE must still round-trip anything.
    for text in UNICODE_PROBES:
        ids = encode_batch(tokenizer, [text])[0]
        assert tokenizer.decode(ids).lstrip(" ") == text


def test_special_token_decoding(tokenizer):
    ids = encode_batch(tokenizer, ["Hi."])[0] + [tokenizer.eos_token_id]
    assert "<|endoftext|>" in tokenizer.decode(ids)
    assert "<|endoftext|>" not in tokenizer.decode(ids, skip_special_tokens=True)


def test_save_reload_gives_identical_ids(tokenizer, stories, tmp_path):
    tokenizer.save_pretrained(tmp_path)
    reloaded = load_tokenizer(tmp_path)
    assert reloaded.clean_up_tokenization_spaces is False
    for text in [*stories[:20], *UNICODE_PROBES]:
        assert encode_batch(reloaded, [text]) == encode_batch(tokenizer, [text])
    assert tokenizer_fingerprint(tmp_path) == tokenizer_fingerprint(tmp_path)


def test_diagnostics(tokenizer, stories):
    diag = tokenizer_diagnostics(tokenizer, stories[:50])
    assert diag["roundtrip_exact_fraction"] == 1.0
    assert diag["characters_per_token"] > 1.0
    assert all(p["roundtrip_exact"] for p in diag["probes"])
    assert diag["single_byte_tokens"] >= 256  # full byte alphabet
