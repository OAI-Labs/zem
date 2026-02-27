#!/usr/bin/env python3
import sys
import re
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

def update_version(new_version: str):
    project_root = Path(__file__).parent.parent
    
    # 1. Update pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        # Dùng \g<1> để tránh ambiguity với version bắt đầu bằng số
        new_content = re.sub(r'(^version = )".*"', r'\g<1>"' + new_version + '"', content, flags=re.MULTILINE)
        pyproject_path.write_text(new_content)
        logger.info(f"Updated {pyproject_path.relative_to(project_root)}")
    
    # 2. Update src/xfmr_zem/__init__.py
    init_path = project_root / "src" / "xfmr_zem" / "__init__.py"
    if init_path.exists():
        content = init_path.read_text()
        new_content = re.sub(r'(^__version__ = )".*"', r'\g<1>"' + new_version + '"', content, flags=re.MULTILINE)
        init_path.write_text(new_content)
        logger.info(f"Updated {init_path.relative_to(project_root)}")
    
    # 3. Update src/xfmr_zem/ui/frontend/package.json
    pkg_path = project_root / "src" / "xfmr_zem" / "ui" / "frontend" / "package.json"
    if pkg_path.exists():
        content = pkg_path.read_text()
        new_content = re.sub(r'("version": )".*"', r'\g<1>"' + new_version + '"', content)
        pkg_path.write_text(new_content)
        logger.info(f"Updated {pkg_path.relative_to(project_root)}")
    
    # 4. Update README.md (Version Badge)
    readme_path = project_root / "README.md"
    if readme_path.exists():
        content = readme_path.read_text()
        # Tìm badge version: [![Version](https://img.shields.io/badge/version-X.Y.Z-blue.svg)]
        new_content = re.sub(r'(badge/version-)\d+\.\d+\.\d+(-blue\.svg)', r'\g<1>' + new_version + r'\g<2>', content)
        readme_path.write_text(new_content)
        logger.info(f"Updated {readme_path.relative_to(project_root)}")

    # 5. Update CHANGELOG.md (Add new header if not exists)
    changelog_path = project_root / "CHANGELOG.md"
    if changelog_path.exists():
        content = changelog_path.read_text()
        header = f"## [{new_version}] - {datetime.now().strftime('%Y-%m-%d')}"
        if header not in content:
            new_content = header + "\n\n### Changed\n- Bump version to " + new_version + "\n\n" + content
            changelog_path.write_text(new_content)
            logger.info(f"Added new version header to {changelog_path.relative_to(project_root)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/bump_version.py X.Y.Z")
        sys.exit(1)
    
    new_ver = sys.argv[1]
    if not re.match(r"^\d+\.\d+\.\d+", new_ver):
        logger.error(f"Invalid version format: {new_ver}. Use X.Y.Z")
        sys.exit(1)
        
    update_version(new_ver)
    logger.success(f"Successfully bumped all files to version {new_ver}")
