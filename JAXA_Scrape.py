"""
Helper script to download and unpack JAXA ALOS World 3D 30 m (AW3D30) DSM tiles.

Requires a free JAXA ALOS World 3D account:
https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm
"""

import argparse
import os
import shutil
import time
import zipfile

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

JAXA_AW3D30_URL = "https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/index.htm"


def scrape(download_folder: str = "ALOS_Data") -> None:
    """Open the JAXA AW3D30 portal in Chrome and download tile archives."""
    download_folder = os.path.abspath(download_folder)
    os.makedirs(download_folder, exist_ok=True)

    options = Options()
    options.add_experimental_option("prefs", {"download.default_directory": download_folder})

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(JAXA_AW3D30_URL)
        WebDriverWait(driver, 120).until(
            EC.presence_of_element_located((By.TAG_NAME, "map"))
        )

        soup = BeautifulSoup(driver.page_source, "html.parser")
        tile_links = [
            "https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/" + area.get("href")
            for area in soup.find_all("area")
            if area.get("href")
        ]

        for link in tile_links:
            driver.get(link)
            download_buttons = WebDriverWait(driver, 60).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "input"))
            )
            for btn in download_buttons:
                try:
                    btn.click()
                    time.sleep(3)
                except Exception as exc:
                    print(f"Error clicking download button on {link}: {exc}")
    finally:
        driver.quit()


def extract_zips(download_folder: str = "ALOS_Data") -> None:
    """Extract all downloaded ``.zip`` archives in ``download_folder`` and remove zips."""
    for filename in os.listdir(download_folder):
        if filename.endswith(".zip"):
            archive_path = os.path.join(download_folder, filename)
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(download_folder)
            os.remove(archive_path)


def extract_dsm(download_folder: str = "ALOS_Data") -> None:
    """Move ``*_DSM.tif`` files out of extracted subfolders into ``download_folder``."""
    for entry in os.listdir(download_folder):
        folder_path = os.path.join(download_folder, entry)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if fname.endswith("_DSM.tif"):
                    shutil.move(
                        os.path.join(folder_path, fname),
                        os.path.join(download_folder, fname),
                    )
            shutil.rmtree(folder_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and unpack ALOS AW3D30 DSM tiles.")
    parser.add_argument(
        "--output-dir",
        default="ALOS_Data",
        help="Destination folder for downloaded *_DSM.tif files (default: ALOS_Data).",
    )
    args = parser.parse_args()

    scrape(args.output_dir)
    extract_zips(args.output_dir)
    extract_dsm(args.output_dir)


if __name__ == "__main__":
    main()
