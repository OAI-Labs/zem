"""
ArgillaConnectionFactory – Singleton factory để quản lý kết nối đến Argilla server.

Nguyên tắc SOLID:
  - S: Chỉ chịu trách nhiệm khởi tạo & cache Argilla client
  - O: Dễ mở rộng thêm auth provider (token / OAuth)
"""
import os
from typing import Optional
from loguru import logger


class ArgillaConnectionFactory:
    """
    Singleton factory quản lý kết nối Argilla.
    Lazy-init client, re-use nếu đã kết nối.
    """
    _instance: Optional[object] = None
    _api_url: Optional[str] = None
    _api_key: Optional[str] = None

    @classmethod
    def get_client(
        cls,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Trả về Argilla client (singleton).
        Ưu tiên: tham số truyền vào > biến môi trường > mặc định localhost.

        Args:
            api_url: URL của Argilla server
            api_key: API key xác thực

        Returns:
            argilla.Argilla instance
        """
        try:
            import argilla as rg
        except ImportError:
            raise ImportError(
                "Thiếu dependency Argilla. "
                "Cài đặt bằng: pip install 'xfmr-zem[argilla]'"
            )

        resolved_url = api_url or os.getenv("ARGILLA_API_URL", "http://localhost:6900")
        resolved_key = api_key or os.getenv("ARGILLA_API_KEY", "argilla.apikey")

        # Tạo mới nếu chưa có hoặc config thay đổi
        if (
            cls._instance is None
            or cls._api_url != resolved_url
            or cls._api_key != resolved_key
        ):
            logger.info(f"Kết nối Argilla tại: {resolved_url}")
            cls._instance = rg.Argilla(
                api_url=resolved_url,
                api_key=resolved_key,
            )
            cls._api_url = resolved_url
            cls._api_key = resolved_key
            logger.info("Argilla client đã sẵn sàng.")

        return cls._instance

    @classmethod
    def reset(cls):
        """Xóa singleton (dùng cho testing)."""
        cls._instance = None
        cls._api_url = None
        cls._api_key = None
