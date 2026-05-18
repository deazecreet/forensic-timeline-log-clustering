from log_clustering.preprocessing import build_clustering_text, clean_message, normalize_source


def test_clean_message_normalizes_paths_hex_timestamps_and_numbers():
    text = (
        r"C:\Users\User\AppData\Local\Temp\evil.exe "
        "2023-12-26T00:20:03 0xABCDEF123456789 port 443"
    )

    cleaned = clean_message(text, stop_words=frozenset())

    assert "path_token" in cleaned
    assert "time_token" in cleaned
    assert "hex_token" in cleaned
    assert "num_token" in cleaned
    assert "abcdef" not in cleaned
    assert "443" not in cleaned


def test_normalize_source_is_stable():
    assert normalize_source("WEBHIST") == "source_webhist"
    assert normalize_source("Windows Event Log") == "source_windows_event_log"
    assert normalize_source("") == "source_unknown"


def test_build_clustering_text_prefixes_source():
    text = build_clustering_text("FILE", "Created C:\\Windows\\System32\\cmd.exe")

    assert text.startswith("source_file ")
    assert "path_token" in text
