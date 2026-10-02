"""feedback.py の回答評価（👍/👎）記録機能のテスト。"""

import json

import pytest

import feedback


def test_record_feedback_appends_jsonl_line(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("質問です", "回答です", feedback.RATING_UP, "thread-a")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["question"] == "質問です"
    assert record["answer"] == "回答です"
    assert record["rating"] == "up"
    assert record["thread_id"] == "thread-a"
    assert "timestamp" in record


def test_record_feedback_appends_multiple_records(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q1", "A1", feedback.RATING_UP, "thread-a")
    feedback.record_feedback("Q2", "A2", feedback.RATING_DOWN, "thread-a")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["rating"] == "up"
    assert json.loads(lines[1])["rating"] == "down"


def test_record_feedback_creates_parent_directory(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_DOWN, "thread-a")

    assert path.exists()


def test_record_feedback_rejects_invalid_rating(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    with pytest.raises(ValueError):
        feedback.record_feedback("Q", "A", "invalid", "thread-a")

    assert not path.exists()


def test_record_feedback_rejects_none_rating(tmp_path, monkeypatch):
    """異常系: ratingにNoneが渡された場合も不正値として拒否する。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    with pytest.raises(ValueError):
        feedback.record_feedback("Q", "A", None, "thread-a")

    assert not path.exists()


def test_record_feedback_allows_empty_question_and_answer(tmp_path, monkeypatch):
    """境界値: 質問・回答が空文字列でも記録自体は成功する（呼び出し側の責務外）。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("", "", feedback.RATING_UP, "thread-a")

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["question"] == ""
    assert record["answer"] == ""


def test_record_feedback_preserves_non_ascii_characters_unescaped(tmp_path, monkeypatch):
    """日本語・絵文字を含む質問/回答が \\uXXXX にエスケープされず、
    人間が読める形式のままJSON Linesとして書き込まれる（ensure_ascii=Falseの確認）。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("日本語の質問🙂", "日本語の回答📝", feedback.RATING_DOWN, "thread-a")

    raw_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "日本語の質問🙂" in raw_line
    assert "日本語の回答📝" in raw_line
    assert "\\u" not in raw_line


def test_record_feedback_appends_without_truncating_existing_content(tmp_path, monkeypatch):
    """境界値: 既に他の内容が書き込まれたファイルに対しても、上書きせず末尾に追記する。"""
    path = tmp_path / "feedback.jsonl"
    path.write_text('{"pre_existing": true}\n', encoding="utf-8")
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_UP, "thread-a")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"pre_existing": True}
    assert json.loads(lines[1])["rating"] == "up"


def test_default_feedback_path_points_to_data_directory():
    """境界値/設定確認: モジュールデフォルトのFEEDBACK_PATHはプロジェクト直下の
    data/feedback.jsonl を指す（.gitignoreの対象パスと一致している必要がある）。"""
    assert feedback.FEEDBACK_PATH.name == "feedback.jsonl"
    assert feedback.FEEDBACK_PATH.parent.name == "data"


def test_record_feedback_skips_duplicate(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_UP, "thread-a")
    feedback.record_feedback("Q", "A", feedback.RATING_DOWN, "thread-a")

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_record_feedback_allows_same_text_in_other_thread(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_UP, "thread-a")
    feedback.record_feedback("Q", "A", feedback.RATING_UP, "thread-b")

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_get_recorded_rating(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    assert feedback.get_recorded_rating("Q", "A", "thread-a") is None
    feedback.record_feedback("Q", "A", feedback.RATING_DOWN, "thread-a")

    assert feedback.get_recorded_rating("Q", "A", "thread-a") == "down"
    assert feedback.get_recorded_rating("Q", "A", "thread-b") is None


def test_get_recorded_rating_skips_corrupted_and_non_dict_lines(tmp_path, monkeypatch):
    """異常系: 破損行・空行・dictでないJSON行が混在していても例外にならず、有効な行を検出する。"""
    path = tmp_path / "feedback.jsonl"
    valid = json.dumps(
        {"thread_id": "thread-a", "question": "Q", "answer": "A", "rating": "up"},
        ensure_ascii=False,
    )
    path.write_text(f'not json\n\n[1, 2]\n"str"\n{valid}\n', encoding="utf-8")
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    assert feedback.get_recorded_rating("Q", "A", "thread-a") == "up"
    assert feedback.get_recorded_rating("Q", "A", "thread-x") is None


def test_get_recorded_rating_returns_none_for_only_corrupted_file(tmp_path, monkeypatch):
    """異常系: 全行が破損していてもNoneを返す。"""
    path = tmp_path / "feedback.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    assert feedback.get_recorded_rating("Q", "A", "thread-a") is None


def test_record_feedback_after_corrupted_line_still_appends(tmp_path, monkeypatch):
    """異常系: 破損行があっても未記録の評価は追記される。"""
    path = tmp_path / "feedback.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_UP, "thread-a")

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert feedback.get_recorded_rating("Q", "A", "thread-a") == "up"


def test_record_feedback_distinguishes_question_and_answer(tmp_path, monkeypatch):
    """境界値: 質問のみ・回答のみが異なる場合は別レコードとして記録される。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_UP, "t")
    feedback.record_feedback("Q2", "A", feedback.RATING_UP, "t")
    feedback.record_feedback("Q", "A2", feedback.RATING_UP, "t")

    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_record_feedback_invalid_rating_rejected_even_if_duplicate(tmp_path, monkeypatch):
    """境界値: 記録済みの組でも不正ratingはValueErrorになる。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)
    feedback.record_feedback("Q", "A", feedback.RATING_UP, "t")

    with pytest.raises(ValueError):
        feedback.record_feedback("Q", "A", "bad", "t")


def test_load_feedback_records_skips_missing_file_and_broken_lines(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)
    assert feedback.load_feedback_records() == []

    path.write_text(
        '{"rating": "up", "question": "Q1"}\nnot json\n{"rating": "bad"}\n[1]\n{"rating": "down", "question": "Q2"}\n',
        encoding="utf-8",
    )

    records = feedback.load_feedback_records()
    assert [r["question"] for r in records] == ["Q1", "Q2"]


def test_summarize_feedback_counts_all_and_recent():
    records = [{"rating": "down"}, {"rating": "up"}, {"rating": "up"}, {"rating": "down"}]

    summary = feedback.summarize_feedback(records, recent_n=2)

    assert summary["all"] == {"up": 2, "down": 2, "total": 4}
    assert summary["recent"] == {"up": 1, "down": 1, "total": 2}


def test_summarize_feedback_empty():
    summary = feedback.summarize_feedback([])

    assert summary["all"] == {"up": 0, "down": 0, "total": 0}
    assert summary["recent"] == {"up": 0, "down": 0, "total": 0}


def test_extract_down_records_newest_first_with_limit():
    records = [
        {"rating": "down", "question": "Q1"},
        {"rating": "up", "question": "Q2"},
        {"rating": "down", "question": "Q3"},
        {"rating": "down", "question": "Q4"},
    ]

    assert [r["question"] for r in feedback.extract_down_records(records)] == ["Q4", "Q3", "Q1"]
    assert [r["question"] for r in feedback.extract_down_records(records, limit=2)] == ["Q4", "Q3"]


def test_summarize_feedback_recent_n_boundaries():
    records = [{"rating": "up"}, {"rating": "down"}]

    assert feedback.summarize_feedback(records, recent_n=0)["recent"] == {"up": 0, "down": 0, "total": 0}
    assert feedback.summarize_feedback(records, recent_n=-1)["recent"] == {"up": 0, "down": 0, "total": 0}
    assert feedback.summarize_feedback(records, recent_n=100)["recent"] == {"up": 1, "down": 1, "total": 2}


def test_extract_down_records_empty_and_no_downs_and_zero_limit():
    assert feedback.extract_down_records([]) == []
    assert feedback.extract_down_records([{"rating": "up"}]) == []
    assert feedback.extract_down_records([{"rating": "down"}], limit=0) == []
