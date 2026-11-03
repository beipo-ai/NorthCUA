"""Historical OSWorld1 text projection; never rewrite tool-call arguments."""

import re

from nexau.core.messages import Message, TextBlock, ToolResultBlock

_X = re.compile(r"(?P<p>[\"']?\bx\b[\"']?\s*[:=]\s*)(?P<v>-?\d+(?:\.\d+)?)", re.I)
_Y = re.compile(r"(?P<p>[\"']?\by\b[\"']?\s*[:=]\s*)(?P<v>-?\d+(?:\.\d+)?)", re.I)
_POS = re.compile(
    r"(?P<prefix>\b(?:at|coordinate|coordinates|position)\s*\(\s*)"
    r"(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)(?P<suffix>\s*\))",
    re.I,
)


def normalize_text(text: str, width: int, height: int) -> str:
    if not text or width <= 0 or height <= 0:
        return text
    text = re.sub(rf"\b{width}\s*(?:x|X|×)\s*{height}\b", "1000 x 1000", text)
    text = _X.sub(lambda m: f"{m['p']}{round(float(m['v']) * 1000 / width)}", text)
    text = _Y.sub(lambda m: f"{m['p']}{round(float(m['v']) * 1000 / height)}", text)
    return _POS.sub(
        lambda m: (
            f"{m['prefix']}{round(float(m['x']) * 1000 / width)}, {round(float(m['y']) * 1000 / height)}{m['suffix']}"
        ),
        text,
    )


def normalize_message(message: Message, width: int, height: int) -> Message:
    def text_block(block):
        return block.model_copy(update={"text": normalize_text(block.text, width, height)})

    blocks = []
    for block in message.content or []:
        if isinstance(block, TextBlock):
            block = text_block(block)
        elif isinstance(block, ToolResultBlock):
            content = block.content
            if isinstance(content, str):
                content = normalize_text(content, width, height)
            elif isinstance(content, list):
                content = [text_block(item) if isinstance(item, TextBlock) else item for item in content]
            block = block.model_copy(update={"content": content})
        blocks.append(block)
    return message.model_copy(update={"content": blocks})
