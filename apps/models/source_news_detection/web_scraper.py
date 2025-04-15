# web_scraper.py
import logging
import re
import requests
from urllib.parse import urlparse
from typing import List, Dict

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy import signals
from scrapy.signalmanager import dispatcher
from scrapy.utils.project import get_project_settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("web_scraper")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TRUSTED_DOMAINS = [
    "wikipedia.org",
    "timesofindia.indiatimes.com",
    "indianexpress.com",
    "nytimes.com",
    "reuters.com",
    "bbc.com",
    "apnews.com",
    "theguardian.com",
    "factcheck.org",
    "snopes.com",
    "politifact.com"
]

class ContentSpider(scrapy.Spider):
    name = "content_spider"
    
    def __init__(self, url=None, *args, **kwargs):
        super(ContentSpider, self).__init__(*args, **kwargs)
        self.start_urls = [url] if url else []
        self.custom_settings = {
            'USER_AGENT': USER_AGENT
        }
    
    def parse(self, response):
        # Extract useful content from the page
        domain = urlparse(response.url).netloc
        
        if "wikipedia.org" in domain:
            paragraphs = response.css("#mw-content-text p").getall()
            title = response.css("#firstHeading::text").get()
        elif "timesofindia.indiatimes.com" in domain:
            paragraphs = response.css(".article_content p, .detail p").getall()
            title = response.css("h1.title::text, h1.heading::text").get()
        elif "indianexpress.com" in domain:
            paragraphs = response.css("article p, .story-details p").getall()
            title = response.css("h1::text").get()
        elif "nytimes.com" in domain:
            paragraphs = response.css("article p").getall()
            title = response.css("h1::text").get()
        elif "bbc.com" in domain:
            paragraphs = response.css("article p").getall()
            title = response.css("h1::text").get()
        else:
            # Generic extraction for other sites
            paragraphs = response.css("p").getall()
            title = response.css("h1::text").get() or response.css("title::text").get()
        
        # Clean HTML tags
        cleaned_paragraphs = []
        for p in paragraphs:
            cleaned = re.sub(r'<.*?>', '', p).strip()
            if cleaned and len(cleaned) > 50:  # Skip short or empty paragraphs
                cleaned_paragraphs.append(cleaned)
        
        content_text = " ".join(cleaned_paragraphs)
        
        # Store the extracted content
        item = {
            "url": response.url,
            "title": title,
            "text": content_text,
            "source": domain,
            "publication_date": self.extract_date(response)
        }
        
        return item
    
    def extract_date(self, response):
        """Extract publication date from various formats"""
        # Try common date meta tags
        date = response.css('meta[property="article:published_time"]::attr(content)').get()
        if not date:
            date = response.css('meta[name="date"]::attr(content)').get()
        if not date:
            date = response.css('time::attr(datetime)').get()
        
        # If date extraction fails, return None
        return date

class GoogleSearchScraper:
    def __init__(self):
        self.headers = {"User-Agent": USER_AGENT}
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        """
        Perform a Google search and return the top result URLs.
        Note: In a production environment, you should use the official Google Search API.
        """
        # This is a simplified implementation. In production, use Google Search API
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        logger.info(f"Searching Google for: {query}")
        try:
            response = requests.get(search_url, headers=self.headers)
            
            # Extract URLs (simplified - real implementation would need more robust parsing)
            # In production, use Google Search API or a specialized library
            urls = re.findall(r'href="(https?://[^"]+)"', response.text)
            
            # Filter for trusted domains
            filtered_urls = []
            for url in urls:
                domain = urlparse(url).netloc
                if any(trusted in domain for trusted in TRUSTED_DOMAINS):
                    filtered_urls.append(url)
                    if len(filtered_urls) >= num_results:
                        break
                        
            return filtered_urls[:num_results]
        except Exception as e:
            logger.error(f"Error during Google search: {e}")
            return []

class ContentScraper:
    def __init__(self):
        self.settings = get_project_settings()
        self.settings.update({
            'LOG_LEVEL': 'ERROR',
            'USER_AGENT': USER_AGENT,
            'ROBOTSTXT_OBEY': False,
            'COOKIES_ENABLED': False,
        })
    
    def scrape_url(self, url: str) -> List[Dict]:
        """Scrape content from a single URL"""
        results = []
        
        def crawler_results(signal, sender, item, response, spider):
            results.append(item)
        
        process = CrawlerProcess(self.settings)
        dispatcher.connect(crawler_results, signal=signals.item_scraped)
        
        # Create a new spider class for each URL
        process.crawl(ContentSpider, url=url)
        process.start()  # Blocks until crawling is finished
        
        return results

    def scrape_multiple_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape content from multiple URLs"""
        all_content = []
        
        for url in urls:
            try:
                logger.info(f"Scraping content from: {url}")
                content = self.scrape_url(url)
                all_content.extend(content)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        return all_content