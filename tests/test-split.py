import re

def find_split_position(text, max_length, priority):
	substring = text[:max_length]
	for delimiter in priority:
		index = substring.rfind(delimiter)
		if index != -1 and index >= max_length / 2:
			return index + len(delimiter)
	return max_length

def close_markdown(text):
	text = text.rstrip()
	closed = []
	# Handle bold (**)
	bold_count = len(re.findall(r'\*\*', text))
	if bold_count & 1:
		closed.append('**')
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += closed[-1]

	# Handle italic (*) - single asterisks not part of bold
	italic_count = len(re.findall(r'(?<!\*)\*(?!\*)', text))
	if italic_count & 1:
		closed.append('*')
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += closed[-1]

	# Handle code blocks (```)
	code_block_count = len(re.findall(r'```', text))
	if code_block_count & 1:
		# Code blocks are special because they can contain Markdown syntax; we need to copy the last detected opening sequence
		last_open = text.rfind('```')
		assert last_open != -1, "Unmatched closing code block detected"
		closed.append(text[last_open:].split("\n", 1)[0] + "\n")
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += "```"

	# Handle inline code (`) - single backticks not part of code block
	inline_code_count = len(re.findall(r'(?<!`)`(?!`)', text))
	if inline_code_count & 1:
		closed.append('`')
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += closed[-1]

	# Handle spoiler tags (||) - double pipes not part of code block
	spoiler_count = len(re.findall(r'(?<!`)\|\|(?!`)', text))
	if spoiler_count & 1:
		closed.append('||')
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += closed[-1]
	return text, "".join(reversed(closed))

def split_text(text, max_length=2000, priority=("\n\n", "\n", "\t", "? ", "! ", ". ", ", ", " "), prefix="", suffix=""):
	chunks = []
	opening = ""
	while text:
		if len(text) <= max_length:
			chunks.append(close_markdown(text)[0])
			break

		for adjusted in range(max_length):
			adjusted_max_length = max(max_length - adjusted, 1)

			split_pos = find_split_position(text, adjusted_max_length, priority)
			current_part = opening + text[:split_pos]
			if not current_part.startswith(prefix):
				current_part = prefix + current_part
			remaining_text = text[split_pos:]

			processed_part, new_opening = close_markdown(current_part)
			if not processed_part.endswith(suffix):
				processed_part += suffix
			if len(processed_part) <= max_length:
				opening = new_opening
				break
		chunks.append(processed_part)
		text = remaining_text
	return chunks

# Example usage:
if __name__ == "__main__":
	sample_text = """**This is a bold header**
Here's some text with Markdown. Let's add a [link](https://example.com) and a code block:
```python
print("Hello World")

# This is a comment
def foo(bar):
	return bar + 1

print(foo(2))
```
We need to split this text properly. Don't split URLs like https://example.com/very-long-url?param=value

# Sample LLM text guaranteed to overflow

> Thinking "A year ago, 60 animals lived in the magical garden: 30 hares, 20 wolves and 10 lions. The number of animals in the garden changes only in three cases: when the wolf eats hare and turns into a lion, when a lion eats a hare and turns into a wolf, and when a lion eats a wolf and turns into a hare. Currently, there are no animals left in the garden that can eat each other. Determine the maximum and minimum number of animals to be left in the garden."...
```
After carefully analyzing the problem, here are the results:

- **Maximum number of animals left**: 40 (all hares).  
- **Minimum number of animals left**: 10 (all wolves or hares).  
```
This happens because each predation event reduces the total number of animals by one, and the final state depends on which transformations are prioritized. If you'd like, I can explain the reasoning in more detail! 🐰🐺🦁

# Sample spam text with spoiler and no natural boundaries
||aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa||
"""

	chunks = split_text(
		sample_text.replace("\r\n", "\n").replace("\f", "\n\n"),
		max_length=100,
	)

	for i, chunk in enumerate(chunks):
		print(f"Chunk {i+1} (length {len(chunk)}):\n{chunk}\n")