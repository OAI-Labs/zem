"""
UserFactory – quản lý users/annotators trên Argilla server.

Chức năng:
  - Tạo user mới (annotator / admin)
  - Lấy danh sách users
  - Cập nhật thông tin user
  - Xóa user
  - Quản lý workspace membership
"""
from typing import Any, Dict, List, Optional
from loguru import logger


# Role hợp lệ trong Argilla
VALID_ROLES = {"admin", "annotator"}


class UserFactory:
    """
    Factory quản lý Users/Annotators trên Argilla.
    """

    # ── Create ───────────────────────────────────────────────────────────────

    @staticmethod
    def create_user(
        client,
        username: str,
        password: str,
        role: str = "annotator",
        first_name: str = "",
        last_name: str = "",
    ):
        """
        Tạo user mới.

        Args:
            client: Argilla client
            username: Tên đăng nhập (unique)
            password: Mật khẩu (min 8 ký tự)
            role: "annotator" | "admin"
            first_name: Tên
            last_name: Họ

        Returns:
            rg.User instance
        """
        try:
            import argilla as rg
        except ImportError:
            raise ImportError("Cài: pip install 'xfmr-zem[argilla]'")

        if role not in VALID_ROLES:
            raise ValueError(f"Role không hợp lệ: '{role}'. Dùng: {VALID_ROLES}")

        user = rg.User(
            username=username,
            password=password,
            role=role,
            first_name=first_name or username,
            last_name=last_name,
            client=client,
        )
        user.create()
        logger.info(f"Đã tạo user '{username}' (role={role})")
        return user

    # ── List ─────────────────────────────────────────────────────────────────

    @staticmethod
    def list_users(client, role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả users.

        Args:
            client: Argilla client
            role_filter: Lọc theo role ("admin" | "annotator" | None)

        Returns:
            List dict thông tin từng user
        """
        users = client.users
        result = []
        for u in users:
            role = u.role.value if hasattr(u.role, "value") else str(u.role)
            if role_filter and role != role_filter:
                continue
            result.append({
                "id": str(u.id) if u.id else None,
                "username": u.username,
                "role": role,
                "first_name": u.first_name or "",
                "last_name": u.last_name or "",
                "inserted_at": str(u.inserted_at) if u.inserted_at else None,
            })

        logger.info(f"Tìm thấy {len(result)} users" + (f" (role={role_filter})" if role_filter else ""))
        return result


    # ── Get ──────────────────────────────────────────────────────────────────

    @staticmethod
    def get_user(client, username: str):
        """
        Lấy thông tin user theo username.

        Args:
            client: Argilla client
            username: Tên đăng nhập

        Returns:
            rg.User hoặc None nếu không tìm thấy
        """
        users = client.users
        for u in users:
            if u.username == username:
                return u
        return None

    # ── Delete ───────────────────────────────────────────────────────────────

    @staticmethod
    def delete_user(client, username: str) -> bool:
        """
        Xóa user theo username.

        Args:
            client: Argilla client
            username: Tên đăng nhập cần xóa

        Returns:
            True nếu xóa thành công
        """
        user = UserFactory.get_user(client, username)
        if user is None:
            raise ValueError(f"User '{username}' không tồn tại.")
        user.delete()
        logger.info(f"Đã xóa user '{username}'")
        return True

    # ── Workspace membership ──────────────────────────────────────────────────

    @staticmethod
    def add_to_workspace(client, username: str, workspace_name: str) -> Dict[str, Any]:
        """
        Thêm user vào workspace.

        Args:
            client: Argilla client
            username: Tên user
            workspace_name: Tên workspace

        Returns:
            {"username": str, "workspace": str, "status": "added"}
        """
        try:
            import argilla as rg
        except ImportError:
            raise ImportError("Cài: pip install 'xfmr-zem[argilla]'")

        user = UserFactory.get_user(client, username)
        if user is None:
            raise ValueError(f"User '{username}' không tồn tại.")

        workspace = client.workspaces(workspace_name)
        if not workspace:
            raise ValueError(f"Workspace '{workspace_name}' không tồn tại.")
        workspace = workspace[0] if isinstance(workspace, list) else workspace

        workspace.add_user(user)
        logger.info(f"Đã thêm '{username}' vào workspace '{workspace_name}'")
        return {"username": username, "workspace": workspace_name, "status": "added"}

    @staticmethod
    def remove_from_workspace(client, username: str, workspace_name: str) -> Dict[str, Any]:
        """
        Xóa user khỏi workspace.

        Args:
            client: Argilla client
            username: Tên user
            workspace_name: Tên workspace

        Returns:
            {"username": str, "workspace": str, "status": "removed"}
        """
        user = UserFactory.get_user(client, username)
        if user is None:
            raise ValueError(f"User '{username}' không tồn tại.")

        workspace = client.workspaces(workspace_name)
        if not workspace:
            raise ValueError(f"Workspace '{workspace_name}' không tồn tại.")
        workspace = workspace[0] if isinstance(workspace, list) else workspace

        workspace.remove_user(user)
        logger.info(f"Đã xóa '{username}' khỏi workspace '{workspace_name}'")
        return {"username": username, "workspace": workspace_name, "status": "removed"}

    # ── Annotator stats per dataset ───────────────────────────────────────────

    @staticmethod
    def annotator_stats(
        client,
        dataset_name: str,
        workspace: str = "admin",
    ) -> List[Dict[str, Any]]:
        """
        Thống kê số lượng annotations theo từng annotator trong dataset.

        Args:
            client: Argilla client
            dataset_name: Tên dataset
            workspace: Workspace

        Returns:
            List[{"annotator": str, "submitted": int, "discarded": int, "pending": int}]
        """
        from collections import defaultdict

        dataset = client.datasets(name=dataset_name, workspace=workspace)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' không tồn tại.")
        dataset = dataset[0]

        records = list(dataset.records(with_responses=True))
        stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"submitted": 0, "discarded": 0, "pending": 0})

        for rec in records:
            if not rec.responses:
                continue
            for resp in rec.responses:
                uid = str(resp.user_id) if resp.user_id else "unknown"
                st = resp.status.value if resp.status else "pending"
                stats[uid][st] = stats[uid].get(st, 0) + 1

        result = [
            {
                "annotator_id": uid,
                **counts,
                "total": sum(counts.values()),
            }
            for uid, counts in sorted(stats.items(), key=lambda x: -sum(x[1].values()))
        ]

        logger.info(f"Thống kê {len(result)} annotators trên dataset '{dataset_name}'")
        return result
