---
description: Release a new version tag for xfmr-zem
---

## Release Workflow cho xfmr-zem

Dùng workflow này để bump version và release tag mới. Đảm bảo tất cả các file chứa thông tin phiên bản được cập nhật đồng bộ.

---

### Bước 1: Xác định version mới
Version hiện tại có thể tìm thấy trong `pyproject.toml`. Xác định version mới theo chuẩn Semantic Versioning (X.Y.Z).

### Bước 2: Bump version trong các file hệ thống

Bạn cần cập nhật version tại **3 vị trí** sau:

1.  **`pyproject.toml`**: Dòng `version = "X.Y.Z"`
2.  **`src/xfmr_zem/__init__.py`**: Biến `__version__ = "X.Y.Z"`
3.  **`src/xfmr_zem/ui/frontend/package.json`**: Trường `"version": "X.Y.Z"`

### Bước 2: Bump version tự động

Sử dụng script `scripts/bump_version.py` để cập nhật đồng bộ các file: `pyproject.toml`, `src/xfmr_zem/__init__.py`, `src/xfmr_zem/ui/frontend/package.json`, và `CHANGELOG.md`.

// turbo
```bash
# Thay X.Y.Z bằng version mới
python scripts/bump_version.py X.Y.Z
```

### Bước 3: Kiểm tra lại (Optional)
Bạn có thể chạy `uv run zem --version` để đảm bảo CLI đã nhận version mới.

### Bước 4: Commit all changes

// turbo
```bash
git add pyproject.toml src/xfmr_zem/__init__.py src/xfmr_zem/ui/frontend/package.json CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
```

### Bước 5: Push main và Tạo Tag

// turbo
```bash
git push origin main
git tag vX.Y.Z
git push origin vX.Y.Z
```

---

### Quy tắc Bump Version (SemVer)
- **PATCH** (0.3.9 → 0.3.10): Sửa lỗi, không thêm tính năng.
- **MINOR** (0.3.10 → 0.4.0): Thêm tính năng mới nhưng vẫn tương thích ngược.
- **MAJOR** (1.0.0): Có thay đổi lớn không tương thích ngược.

**Lưu ý:** CLI sẽ tự động lấy version từ `src/xfmr_zem/__init__.py`, do đó chỉ cần đảm bảo file này được cập nhật chính xác.
