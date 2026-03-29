from pathlib import Path

import load_data


def test_load_sms_spam_parses_tsv(tmp_path: Path) -> None:
    p = tmp_path / "SMSSpamCollection"
    p.write_text(
        "ham\tHello there\n"
        "spam\tWin cash now!!!\n"
        "ham\tSee you later\n",
        encoding="utf-8",
    )
    df = load_data.load_sms_spam(p)
    assert list(df.columns) == ["label", "text"]
    assert df["label"].tolist() == [0, 1, 0]
    assert "Win cash" in df.loc[1, "text"]


def test_load_sms_spam_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    try:
        load_data.load_sms_spam(missing)
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")
