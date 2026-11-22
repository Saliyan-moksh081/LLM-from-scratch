from scripts.prepare_tiny_data import split_stories


def test_split_stories_keeps_complete_stories() -> None:
    """Split the TinyStories document marker into complete stories."""
    raw_text = "First story<|endoftext|>Second story<|endoftext|>"

    assert split_stories(raw_text) == ["First story", "Second story"]


def test_split_stories_discards_truncated_final_story() -> None:
    """Discard the last story when the downloaded text ends before its marker."""
    raw_text = "Complete story<|endoftext|>Truncated story"

    assert split_stories(raw_text) == ["Complete story"]
    