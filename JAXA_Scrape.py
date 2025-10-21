from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import requests
from bs4 import BeautifulSoup
import time
import os
import zipfile
import shutil

# If the package imports fail, try using pip to install selenium and the other packages

# PACKAGE REFERENCE
# selenium: allows for opening and interacting with a browser automatically
# time: allows for time delays for browser interaction
# requests and BeautifulSoup: allows for parsing HTML and downloading files from websites
# os, zipfile, shutil: allows for interacting with directories and handling zip files


download_folder = "/Users/williamlee/Desktop/MIST_Maps/ALOS_Data" 
# ^ CHANGE THIS TO DOWNLOAD FOLDER OF YOUR PREFERENCE



def scrape():

    options = Options()
    prefs = {"download.default_directory": download_folder}
    options.add_experimental_option("prefs", prefs)


    driver = webdriver.Chrome(options=options)
    driver.get("https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/index.htm")


    map_frame = WebDriverWait(driver, 120).until(
        EC.presence_of_element_located((By.TAG_NAME, "map"))
    )

    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    tile_links = []

    for area in soup.find_all("area"):
        href = area.get("href")
        link = "https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/" + href
        tile_links.append(link)

    for link in tile_links:
        driver.get(link)
        download_buttons = WebDriverWait(driver, 60).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "input"))
        )
        for btn in download_buttons:
            try:
                btn.click()
                time.sleep(3)  # wait for download to start
            except Exception as e:
                print(f"Error clicking button: {e}")
    driver.quit()

def open():
    for filename in os.listdir(download_folder):
        if filename.endswith(".zip"):
            file = os.path.join(download_folder, filename)

            with zipfile.ZipFile((file), 'r') as zip_ref:
                zip_ref.extractall(download_folder)
                os.remove(file)
            

def extract_dsm():
    for folder in os.listdir(download_folder):
        folder_path = os.path.join(download_folder, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("_DSM.tif"):
                    file_path = os.path.join(folder_path, file)
                    dest_path = os.path.join(download_folder, file)
                    shutil.move(file_path, dest_path)
            shutil.rmtree(folder_path)

if __name__ == "__main__":
    scrape()
    open()
    extract_dsm()