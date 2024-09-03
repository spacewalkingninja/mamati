import eel

# Assuming spot.market.ticker_price(symbol=SYMBOL) is the function to get the current price
# and SYMBOL is the asset symbol you want to monitor.

class PriceMonitor:
    def __init__(self, symbol, spot):
        self.symbol = symbol
        self.current_price = None
        self.spot = spot
        self.stop_event = False

    def _monitor_price(self):
        while not self.stop_event:
            try:
                rawdata = self.spot.market.ticker_price(symbol=self.symbol)
                self.current_price = rawdata[0]['price']
                price = rawdata[0]['price']
                
                # Add a sleep interval to avoid excessive API calls
                eel.sleep(0.1)
            except Exception as e:
                print(f"Error fetching price: {e}")
                # Sleep for a while before retrying in case of error
                eel.sleep(10)

    def start(self):
        eel.spawn(self._monitor_price)

    def stop(self):
        self.stop_event = True

    def get_price(self):
        return self.current_price
