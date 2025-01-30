import pytest
from misc import util
split_across = util.split_across

def test_basic_split():
    # Test basic string splitting
    assert split_across("Long text here", lim=10) == ["Long text", "here"]
    assert split_across("Short", lim=10) == ["Short"]

def test_with_prefix_suffix():
    # Test with prefix and suffix
    assert split_across("Hello World", prefix="Pre: ", suffix=" :End", lim=15) == ["Pre: Hello :End", "Pre: World :End"]
    assert split_across("Test", prefix="<", suffix=">", lim=10) == ["<Test>"]

def test_code_blocks():
    # Test code block handling
    assert split_across("```code\nstuff```", prefix="Pre: ") == ["Pre: ```code\nstuff```"]
    assert split_across("```python\nprint('hello')\n```\nSome text", lim=20) == [
        "```python\nprint('hello')\n```",
        "Some text"
    ]
    # Test unclosed code block
    assert split_across("```python\nprint('hi')", lim=30) == ["```python\nprint('hi')```"]

def test_bypass():
    # Test bypass functionality
    assert split_across(
        "!command text",
        prefix="Pre: ",
        bypass=(("!",), ()),
        lim=15
    ) == ["!command text"]

    assert split_across(
        "text <end>",
        suffix=" :End",
        bypass=((), ("<end>",)),
        lim=15
    ) == ["text <end>"]

def test_natural_boundaries():
    # Test splitting on natural boundaries
    text = "First sentence. Second sentence! Third? Fourth\nFifth\n\nSixth"
    segments = split_across(text, lim=20)
    
    assert len(segments) > 1
    assert "First sentence" in segments[0]
    assert "Sixth" in segments[-1]

def test_edge_cases():
    # Test empty string
    assert split_across("", lim=10) == [""]
    
    # Test string shorter than limit 
    assert split_across("abc", lim=10) == ["abc"]
    
    # Test exact limit
    assert split_across("exactly10ch", lim=10) == ["exactly10ch"]
    
    # Test very small limit
    segments = split_across("test text", lim=3) 
    assert all(len(s) <= 3 for s in segments)

def test_mode_tlen():
    # Test token length mode
    text = "This is a test of token counting mode"
    segments = split_across(text, lim=5, mode="tlen")
    
    # Each segment should have 5 or fewer tokens
    assert all(len(split_across(s, mode="tlen")) <= 5 for s in segments)

def test_invalid_mode():
    # Test invalid mode raises error
    with pytest.raises(NotImplementedError):
        split_across("test", mode="invalid")

for k, v in tuple(globals().items()):
	if k.startswith("test_") and callable(v):
		print(v())