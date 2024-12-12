import os, time, requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

query = "people smoking"
file_path = os.path.join("d:\SELENIUM\img_{}".format(query))
os.makedirs(file_path, exist_ok=True)
search_url = "https://www.google.com/search?q={}&tbm=isch".format(query)

driver = webdriver.Chrome()
driver.maximize_window()

driver.get(search_url)
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "img.YQ4gaf")))

img_elements = []  # images to be saved 
img_del = []  # small icons (to be deleted)
prev = -1
while(len(img_elements)-len(img_del) != prev):
    prev = len(img_elements)-len(img_del)
    img_elements = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")
    img_del = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf.zr758c")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(10)
for i in img_del:
    img_elements.remove(i)
print(len(img_elements))

for i, element in enumerate(img_elements):
    try:
        element.click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'img.sFlh5c.FyHeAf.iPVvYb')))
        img_url = driver.find_element(By.CSS_SELECTOR, 'img.sFlh5c.FyHeAf.iPVvYb').get_attribute("src")
        response = requests.get(img_url, timeout=5)
        if response.ok:
            f = open(os.path.join(file_path, f"{query}_{i}.jpg"), 'wb')
            f.write(response.content) 
    except Exception:
        print("FAILED TO SAVE IMAGE {}".format(i))

driver.quit()