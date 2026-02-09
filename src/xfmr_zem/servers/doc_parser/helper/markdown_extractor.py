import re

class MarkdownExtractor:
    """
    Class to support extracting text from Markdown components,
    allowing text modification (e.g., spelling correction) and reconstructing Markdown.
    """
    def __init__(self):
        # Regex patterns to identify inline components that need protection
        self.inline_code_pattern = re.compile(r'(`[^`]+`)')
        self.link_pattern = re.compile(r'(\[[^\]]+\]\([^)]+\))')
        self.image_pattern = re.compile(r'(!\[[^\]]*\]\([^)]+\))')
        self.html_tag_pattern = re.compile(r'(<[^>]+>)')
        self.url_pattern = re.compile(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)')

    def _process_segment(self, text, corrector_func):
        """
        Process a text segment: protect special structures and correct the remaining text.
        """
        if not text or not text.strip():
            return text

        # List of elements to protect (do not correct)
        protected_segments = []
        
        def protect_match(match):
            protected_segments.append(match.group(0))
            return f"__PROTECTED_{len(protected_segments)-1}__"

        # 1. Protect URL, Link, Image, Code, HTML
        # Important order: Code -> Image/Link -> URL -> HTML
        temp_text = self.inline_code_pattern.sub(protect_match, text)
        temp_text = self.image_pattern.sub(protect_match, temp_text)
        temp_text = self.link_pattern.sub(protect_match, temp_text)
        temp_text = self.url_pattern.sub(protect_match, temp_text)
        temp_text = self.html_tag_pattern.sub(protect_match, temp_text)

        # 2. Correct spelling on the remaining text
        # Only call corrector if there is meaningful text left and not entirely protected
        if temp_text.strip() and not re.match(r'^__PROTECTED_\d+__$', temp_text.strip()):
             corrected_text = corrector_func(temp_text)
        else:
             corrected_text = temp_text

        # 3. Restore protected elements
        for i, segment in enumerate(protected_segments):
            corrected_text = corrected_text.replace(f"__PROTECTED_{i}__", segment)

        return corrected_text

    def extract_and_correct(self, markdown_content, corrector_func):
        """
        Main function: Iterate through each markdown line, identify components, and apply correction.
        """
        lines = markdown_content.split('\n')
        result_lines = []
        in_code_block = False

        for line in lines:
            # 1. Handle Code Block (```) - Keep content inside unchanged
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            if in_code_block:
                result_lines.append(line)
                continue

            # 2. Skip empty lines or horizontal rules (---, ***)
            if not line.strip() or re.match(r'^[-*_]{3,}$', line.strip()):
                result_lines.append(line)
                continue

            # 3. Handle Markdown components line by line
            
            # Headers (#, ##, ...)
            header_match = re.match(r'^(#{1,6}\s+)(.*)', line)
            if header_match:
                prefix, content = header_match.groups()
                corrected = self._process_segment(content, corrector_func)
                result_lines.append(f"{prefix}{corrected}")
                continue

            # Blockquotes (>)
            blockquote_match = re.match(r'^(\s*>\s+)(.*)', line)
            if blockquote_match:
                prefix, content = blockquote_match.groups()
                corrected = self._process_segment(content, corrector_func)
                result_lines.append(f"{prefix}{corrected}")
                continue

            # List items (Unordered: -, *, + / Ordered: 1., 2.)
            list_match = re.match(r'^(\s*(?:[-*+]|\d+\.)\s+)(.*)', line)
            if list_match:
                prefix, content = list_match.groups()
                corrected = self._process_segment(content, corrector_func)
                result_lines.append(f"{prefix}{corrected}")
                continue

            # Tables (| ... |)
            # Preliminary check if it is a table row
            if '|' in line and re.match(r'^\s*\|.*\|\s*$', line):
                parts = line.split('|')
                corrected_parts = []
                for part in parts:
                    # Skip table header separator line (---)
                    if re.match(r'^\s*:?-+:?\s*$', part):
                        corrected_parts.append(part)
                    else:
                        corrected_parts.append(self._process_segment(part, corrector_func))
                result_lines.append('|'.join(corrected_parts))
                continue

            # 4. Normal text lines (Paragraph)
            result_lines.append(self._process_segment(line, corrector_func))

        return '\n'.join(result_lines)
