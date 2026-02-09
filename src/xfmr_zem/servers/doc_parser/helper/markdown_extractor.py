import re

class MarkdownExtractor:
    """
    Class hỗ trợ trích xuất text từ các thành phần Markdown,
    cho phép chỉnh sửa text (ví dụ: sửa lỗi chính tả) và tái tạo lại Markdown.
    """
    def __init__(self):
        # Regex patterns để nhận diện các thành phần nội tuyến (inline) cần bảo vệ
        self.inline_code_pattern = re.compile(r'(`[^`]+`)')
        self.link_pattern = re.compile(r'(\[[^\]]+\]\([^)]+\))')
        self.image_pattern = re.compile(r'(!\[[^\]]*\]\([^)]+\))')
        self.html_tag_pattern = re.compile(r'(<[^>]+>)')
        self.url_pattern = re.compile(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)')

    def _process_segment(self, text, corrector_func):
        """
        Xử lý một đoạn text: bảo vệ các cấu trúc đặc biệt và sửa lỗi phần text còn lại.
        """
        if not text or not text.strip():
            return text

        # Danh sách các phần tử cần bảo vệ (không sửa lỗi)
        protected_segments = []
        
        def protect_match(match):
            protected_segments.append(match.group(0))
            return f"__PROTECTED_{len(protected_segments)-1}__"

        # 1. Bảo vệ URL, Link, Image, Code, HTML
        # Thứ tự quan trọng: Code -> Image/Link -> URL -> HTML
        temp_text = self.inline_code_pattern.sub(protect_match, text)
        temp_text = self.image_pattern.sub(protect_match, temp_text)
        temp_text = self.link_pattern.sub(protect_match, temp_text)
        temp_text = self.url_pattern.sub(protect_match, temp_text)
        temp_text = self.html_tag_pattern.sub(protect_match, temp_text)

        # 2. Sửa lỗi chính tả trên phần text còn lại
        # Chỉ gọi corrector nếu còn text có ý nghĩa và không phải toàn bộ là protected
        if temp_text.strip() and not re.match(r'^__PROTECTED_\d+__$', temp_text.strip()):
             corrected_text = corrector_func(temp_text)
        else:
             corrected_text = temp_text

        # 3. Khôi phục các phần tử đã bảo vệ
        for i, segment in enumerate(protected_segments):
            corrected_text = corrected_text.replace(f"__PROTECTED_{i}__", segment)

        return corrected_text

    def extract_and_correct(self, markdown_content, corrector_func):
        """
        Hàm chính: Duyệt qua từng dòng markdown, xác định thành phần, và áp dụng sửa lỗi.
        """
        lines = markdown_content.split('\n')
        result_lines = []
        in_code_block = False

        for line in lines:
            # 1. Xử lý Code Block (```) - Giữ nguyên nội dung bên trong
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            if in_code_block:
                result_lines.append(line)
                continue

            # 2. Bỏ qua dòng trống hoặc dòng kẻ ngang (---, ***)
            if not line.strip() or re.match(r'^[-*_]{3,}$', line.strip()):
                result_lines.append(line)
                continue

            # 3. Xử lý các thành phần Markdown theo dòng
            
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
            # Kiểm tra sơ bộ xem có phải dòng bảng không
            if '|' in line and re.match(r'^\s*\|.*\|\s*$', line):
                parts = line.split('|')
                corrected_parts = []
                for part in parts:
                    # Bỏ qua dòng phân cách tiêu đề bảng (---)
                    if re.match(r'^\s*:?-+:?\s*$', part):
                        corrected_parts.append(part)
                    else:
                        corrected_parts.append(self._process_segment(part, corrector_func))
                result_lines.append('|'.join(corrected_parts))
                continue

            # 4. Các dòng văn bản thông thường (Paragraph)
            result_lines.append(self._process_segment(line, corrector_func))

        return '\n'.join(result_lines)
