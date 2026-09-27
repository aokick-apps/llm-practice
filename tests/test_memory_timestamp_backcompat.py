"""会話ログのタイムスタンプ（マイクロ秒精度対応）に関する追加の正常系・異常系・境界値テスト。

tests/test_memory.py に対する補完として、以下の観点を実際の save_conversation() 呼び出し
（時刻をモックしない実時間ベース）とファイル名の直接検証で確認する。
- 旧形式（秒精度のみ、マイクロ秒無し）ファイル名の後方互換読み込み
- 同一秒内に複数回保存した場合のファイル名一意性・list_threads()/load_conversation()の順序
- パース不能なファイル名に対するフォールバックの安全性
"""

from datetime import datetime

import memory


def test_save_conversation_multiple_calls_within_same_second_produce_unique_filenames(tmp_path, monkeypatch):
    """正常系: ループで連続呼び出しても（同一秒内に収まっても）ファイル名は衝突しない。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    paths = [memory.save_conversation(f"質問{i}", f"回答{i}", thread_id="thread-a") for i in range(20)]

    assert len(paths) == len(set(paths))
    assert memory.conversation_count("thread-a") == 20


def test_load_conversation_order_matches_save_order_for_rapid_successive_calls(tmp_path, monkeypatch):
    """正常系: 同一スレッドへ連続保存した場合、load_conversation()の順序は保存順と一致する。

    マイクロ秒精度が無ければ同一秒内の並びが不定になりうる観点を、
    実際のdatetime.now()を使った呼び出しで確認する（時刻のモックはしない）。
    """
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    questions = [f"質問{i}" for i in range(15)]
    for q in questions:
        memory.save_conversation(q, f"回答:{q}", thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert [c["question"] for c in conversations] == questions


def test_list_threads_order_matches_reverse_save_order_for_rapid_successive_threads(tmp_path, monkeypatch):
    """正常系: 複数スレッドを連続で保存した場合、list_threads()は保存順の逆（新しい順）になる。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    thread_ids = [f"thread-{i}" for i in range(10)]
    for tid in thread_ids:
        memory.save_conversation("Q", "A", thread_id=tid)

    threads = memory.list_threads()

    assert [t["thread_id"] for t in threads] == list(reversed(thread_ids))


def test_parse_created_at_reads_legacy_second_precision_filename(tmp_path):
    """後方互換性: マイクロ秒を含まない旧形式ファイル名（15文字の秒精度タイムスタンプ）を
    正しく読み込める（micro秒は0になる）。"""
    path = tmp_path / "20240315_134522_ab12cd_質問.md"
    path.write_text("dummy", encoding="utf-8")

    assert memory._parse_created_at(path) == datetime(2024, 3, 15, 13, 45, 22)


def test_parse_created_at_prefers_new_microsecond_format_over_legacy(tmp_path):
    """正常系: マイクロ秒付きの新形式ファイル名は、旧形式へフォールバックせず新形式で解釈される。"""
    path = tmp_path / "20240315_134522_u654321_ab12cd_質問.md"
    path.write_text("dummy", encoding="utf-8")

    assert memory._parse_created_at(path) == datetime(2024, 3, 15, 13, 45, 22, 654321)


def test_parse_created_at_unparseable_filename_falls_back_to_mtime_without_raising(tmp_path):
    """異常系: 命名規則に一切合わない極端なファイル名でも例外を出さずmtimeにフォールバックする。"""
    path = tmp_path / "!!!invalid###.md"
    path.write_text("dummy", encoding="utf-8")

    result = memory._parse_created_at(path)

    assert result == datetime.fromtimestamp(path.stat().st_mtime)


def test_parse_created_at_empty_filename_prefix_falls_back_to_mtime_without_raising(tmp_path):
    """境界値: ファイル名が命名規則より極端に短い場合も例外を出さずmtimeにフォールバックする。"""
    path = tmp_path / "a.md"
    path.write_text("dummy", encoding="utf-8")

    result = memory._parse_created_at(path)

    assert result == datetime.fromtimestamp(path.stat().st_mtime)


def test_parse_created_at_legacy_filename_with_all_digit_uuid_suffix_is_not_misparsed_as_new_format(tmp_path):
    """境界値: 旧形式ファイル名（15文字タイムスタンプ + "_" + uuid hexの先頭6文字）は、
    そのuuid接尾辞がたまたま数字6桁のみであっても、新形式のマイクロ秒マーカー
    （"_u<6桁の数字>"）と混同されず、マイクロ秒0の旧形式として正しく解釈される。
    """
    legacy_filename_with_digit_only_uuid_suffix = "20240315_134522_654321_質問.md"
    path = tmp_path / legacy_filename_with_digit_only_uuid_suffix
    path.write_text("dummy", encoding="utf-8")

    result = memory._parse_created_at(path)

    assert result == datetime(2024, 3, 15, 13, 45, 22)
