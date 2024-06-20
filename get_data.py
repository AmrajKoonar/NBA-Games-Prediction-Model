# #Imports:
# import os #comes with python
# from bs4 import BeautifulSoup #for webscraping
# from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout #allows code to open a web browser for grabbing HTMLs
# import time #comes with python

# #all the seasons we are doing:
# SEASONS = list(range(2016, 2024)) #2016 2017 2018 2019 2020 2021 2022 2023

# #directories to store data:
# DATA_DIR = "data" 

# #directory inside directory:
# STANDINGS_DIR = os.path.join(DATA_DIR, "standings") #os.path.join gives us pointers to the directors to join them together (join data and standings)
# SCORES_DIR = os.path.join(DATA_DIR, "scores")

# #function to grab html from page when given url and selector
# #async function runs asyncresly, cause we are opening a web browser at the same time
# async def get_html(url, selector, sleep=5, retries=3):
#     html = None
#     #if scraping fails, we retry it 3 times 
#     for i in range(1, retries+1):
#         #makes it so we dont scrape too fast, websites will ban if too fast, so if it fails it gets slower and slower to avoid bans
#         time.sleep(sleep * i)

#         try:
#             #inalizes our playwright instance for us
#             async with async_playwright() as p:

#                 #launches a browser (can be chromium too), wait until its done launching, then contiune running our code
#                 browser = await p.firefox.launch()
#                 #creating a new tab in our browser
#                 page = await browser.new_page()
#                 #sends browsers in that tab to a given page, wait until its finished
#                 await page.goto(url)
#                 #shows our progress
#                 print(await page.title())
#                 #dont want to grab all html of page, so we use selector to grab pieces
#                 html = await page.inner_html(selector)
#         except PlaywrightTimeout: #when theres an error webscraping, print it so we can see it in our logs
#             print(f"Timeout error on {url}")
#             continue #goes back to top of loop then tries again
#         else: #sucessfull scrape then break loop
#             break
#     #returns None if 3 retries
#     return html

# #pass in a season number, scrapes all box scores in season
# async def scrape_season(season):
#     #sets url to the website link needed
#     url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
#     #since async function, use await. set selector to .filter because each month has that 
#     html = await get_html(url, "#content .filter")

#     #use BeautifulSoup class
#     soup = BeautifulSoup(html)
#     #finds all of a certain tag; ("a") for us.
#     links = soup.find_all("a")

#     href = [l["href"] for l in links]
#     standings_pages = [f"https://basketball-reference.com{l}" for l in href]

#     #loop through all standings pages
#     for url in standings_pages:
#         #create path to where we wanna save it
#         save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
#         if os.path.exists(save_path):
#             continue
        
#         #get tables that has the box scores in it from indivual standings pages 
#         html = await get_html(url, "#all_schedule")
#         #save our html, open file in write mode.
#         with open(save_path, "w+") as f:
#             f.write(html)

#     for season in SEASONS:
#         await scrape_season(season)


# standings_files = os.listdir(STANDINGS_DIR)

# async def scrape_game(standings_file):
#     with open(standings_files, 'r') as f:
#         html = f.read()

#     soup = BeautifulSoup(html)
#     links = soup.find_all("a")
#     hrefs = [l.get("href") for l in links]
#     box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
#     box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]

#     for url in box_scores:
#         save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
#         if os.path.exists(save_path):
#             continue

#         html = await get_html(url, "#content")
#         if not html:
#             continue
#         with open(save_path, "w+") as f:
#             f.write(html)


#     for f in standings_files:
#         filepath = os.path.join(STANDINGS_DIR, f)

#         await scrape_game(filepath)


import os
import time
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import asyncio

SEASONS = list(range(2016, 2023))  # 2016, 2017, 2018, 2019, 2020, 2021, 2022

DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")

if not os.path.exists(STANDINGS_DIR):
    os.makedirs(STANDINGS_DIR)

if not os.path.exists(SCORES_DIR):
    os.makedirs(SCORES_DIR)

async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries+1):
        time.sleep(sleep * i)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html

async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]
    
    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue
        
        html = await get_html(url, "#all_schedule")
        with open(save_path, "w+", encoding='utf-8') as f:
            f.write(html)

async def scrape_game(standings_file):
    with open(standings_file, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+", encoding='utf-8') as f:
            f.write(html)

async def main():
    for season in SEASONS:
        await scrape_season(season)

    standings_files = os.listdir(STANDINGS_DIR)
    for season in SEASONS:
        files = [s for s in standings_files if str(season) in s]
        
        for f in files:
            filepath = os.path.join(STANDINGS_DIR, f)
            await scrape_game(filepath)

# Run the main function using asyncio
asyncio.run(main())
