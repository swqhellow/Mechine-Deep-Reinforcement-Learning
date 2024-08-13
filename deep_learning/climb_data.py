import os
import time
import requests
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager

def download_images(url, save_folder, max_images=50, min_height=500):
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Set up Selenium WebDriver for Edge
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')  # Run in headless mode
    driver_path = EdgeChromiumDriverManager().install()
    driver = webdriver.Edge(service=Service(driver_path), options=options)

    # Navigate to the URL
    driver.get(url)
    time.sleep(3)  # Give time for the page to load

    # Scroll down to load more images
    for _ in range(5):  # Adjust the range if needed to load more images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for images to load

    # Find all image elements
    img_elements = driver.find_elements(By.XPATH, '//img')

    # Log the number of images found
    print(f"Found {len(img_elements)} images on the page.")

    downloaded_count = 0

    # Download each image
    for idx, img_element in enumerate(img_elements):
        if downloaded_count >= max_images:
            break

        img_url = img_element.get_attribute('src')
        if not img_url:
            continue

        # Handle data-src attribute if src is not available
        if 'data-src' in img_element.get_attribute('outerHTML'):
            img_url = img_element.get_attribute('data-src')

        # Log the image URL
        print(f"Image {idx + 1}: {img_url}")

        try:
            img_response = requests.get(img_url)
            img_response.raise_for_status()

            # Check image height using Pillow
            img = Image.open(BytesIO(img_response.content))
            if img.height < min_height:
                print(f"Skipping image {idx + 1} (height {img.height}px < {min_height}px)")
                continue

            # Get the image name
            img_name = f'image_{downloaded_count + 1}.jpg'
            img_path = os.path.join(save_folder, img_name)
            with open(img_path, 'wb') as img_file:
                img_file.write(img_response.content)

            print(f'Downloaded {img_name} from {img_url}')
            downloaded_count += 1

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {img_url}: {e}")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")

    # Close the WebDriver
    driver.quit()

# Example usage
classes = ['car','computer','tree']
for i in range(len(classes)):
    url = f'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&dyTabStr=MCwzLDEsMiwsNyw2LDUsMTIsOQ%3D%3D&word={classes[i]}'
    save_folder = f'downloaded_images/{classes[i]}'
    download_images(url, save_folder, max_images=200, min_height=500)


