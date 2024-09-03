import eel

import requests
# Assuming spot.market.ticker_price(symbol=SYMBOL) is the function to get the current price
# and SYMBOL is the asset symbol you want to monitor.
from http.cookiejar import CookieJar
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
import gzip
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from seleniumbase import BaseCase
from seleniumwire import webdriver as wiredriver  # Import from seleniumwire
from fake_useragent import UserAgent
import json
import time 

class LiquidityMonitor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.current_map = None
        self.maps={}
        self.stop_event = False
        
        #self._firefox_driver = webdriver.Firefox()
        #ua = UserAgent(browsers=['edge', 'chrome'])
        
        # Setup Firefox WebDriver
        firefox_options = Options()
        #firefox_options.headless = True  # Uncomment this line for headless mode
        firefox_options.add_argument("--no-sandbox")
        firefox_options.add_argument("--no-sandbox")
        firefox_options.add_argument("--disable-dev-shm-usage")
        firefox_options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0")
        #firefox_options.set_preference("general.useragent.override", ua.random)
        firefox_options.set_preference("accept.untrusted.certs", True)
        firefox_options.set_preference("dom.webdriver.enabled", False)
        firefox_options.set_preference('useAutomationExtension', False)
        firefox_options.set_preference("dom.webdriver.enabled", False)
        firefox_options.set_preference('useAutomationExtension', False)
        firefox_options.add_argument("start-maximized")
        #firefox_options.set_preference("excludeSwitches", ["enable-automation"])
        firefox_options.set_preference('devtools.jsonview.enabled', False)

        #profile_path = os.path.join("C:/", "Users", "Kristian", "AppData", "Roaming", "Mozilla","Firefox","Profiles","p2")
        #firefox_options.add_argument("-profile")
        #firefox_options.add_argument(profile_path)
        #firefox_options.set_preference('profile', profile_path)
        #firefox_options.set_preference('moz:profile', profile_path)
        self.driver = wiredriver.Firefox(options=firefox_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
      


    def _fetch_liquidity_heatmap(self):
        return True

    def _monitor_liquidity(self):
        while not self.stop_event:
            try:
                data = self._fetch_liquidity_heatmap()
                
            except Exception as e:
                print(f"Liquidity fetching not implemented in this version! \n Alternative is to try implementing it yourself using coinglass api ")
                self.current_map = None
                # Sleep for a while before retrying in case of error
                eel.sleep(120)

    def start(self):
        eel.spawn(self._monitor_liquidity)

    def stop(self):
        self.stop_event = True

    def get_map(self):
        return self.current_map
