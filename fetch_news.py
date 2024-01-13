import json
import asyncio
from playwright.async_api import async_playwright
from scrapy.selector import Selector

async def scrape_news():
    async with async_playwright() as p:
        # Initialize browser
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.60secondsnow.com/india/")

        # Wait for the page to load and scroll
        await asyncio.sleep(3)
        await page.evaluate("window.scrollBy(0, 1000)")
        await asyncio.sleep(3)

        # Extract the page content
        html_content = await page.content()
        selector = Selector(text=html_content)

        # Scrape data
        news_data = []
        articles = selector.xpath('//div[contains(@class,"listingpage")]/article')
        for article in articles:
            heading = article.xpath('.//div[contains(@class,"article-content")]/h2/text()').get()
            description = article.xpath('.//div[contains(@class,"article-desc")]/text()').get()
            link = article.xpath('.//div[contains(@class,"article-provider")]/a/@href').get()
            # Only add data if all parts are present
            if heading and description and link:
                news_data.append({
                    "heading": heading.strip(),
                    "description": description.strip(),
                    "link": link.strip()
                })

        # Close the browser
        await browser.close()

        # Save data to a JSON file
        with open('news_india.json', 'w') as file:
            json.dump(news_data, file, indent=4)

        # print(news_data)
        
        return news_data

# Run the scraper

