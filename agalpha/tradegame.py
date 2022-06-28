from cmath import nan
from enum import Enum
import multiprocessing
from matplotlib.font_manager import win32InstalledFonts
from pandas import DataFrame
import numpy as np
from numpy import interp

NUM_MEMORIES = 10

class TradeAction(Enum):
    BUY = 0
    SELL = 1
    DO_NOTHING = 2

class TradeActionException(Exception):
    pass

class TradeGame:
    def __init__(self, df: DataFrame, wallet=100):
        self.df = df
        self.current_step = 0
        self.wallet = wallet # money that we have in $
        self.initial_wallet = wallet
        self.tokens = 0 # tokens we have 
        self.prev_memory = np.zeros(NUM_MEMORIES)

        while np.isnan(self.df["i_macd"].iloc[self.current_step]):
            self.current_step += 1
    
    def step(self, predictor):
        self.current_step += 1

        outputs = self.predict_nn(predictor)
        action = TradeAction(np.argmax(outputs[:2]))
        self.do_action(action) 

    def predict_nn(self, predictor):
        cs = self.current_step
        def inter(val):
            return interp(val, [0, 5000], [-2, 2])

        inputs = [
            self.df["open"].iloc[cs],
            self.df["close"].iloc[cs],
            self.df["high"].iloc[cs],
            self.df["low"].iloc[cs],
            self.df["volume"].iloc[cs],
        ] 

        inputs = inter(inputs).tolist()

        inputs.extend([
            self.df["i_macd"].iloc[cs],
            self.df["i_rsi"].iloc[cs],
            self.df["i_awesome"].iloc[cs],
            self.df["i_stochastic_rsi_d"].iloc[cs],
            self.df["i_stochastic_rsi_k"].iloc[cs],
            self.df["i_bollinger_h"].iloc[cs],
            self.df["i_bollinger_l"].iloc[cs]
        ])

        inputs.extend(self.prev_memory)
        outputs = predictor(inputs)[0]
        self.prev_memory = outputs[-NUM_MEMORIES:] # set last N elements as prev_memory 

        return outputs[:-NUM_MEMORIES].detach().numpy()

    def current_token_price(self):
        return self.df["close"].iloc[self.current_step]

    def fitness(self):
        return self.wallet + (self.tokens * self.current_token_price()) - self.initial_wallet

    def is_done(self):
        return self.current_step == len(self.df["time"]) - 1 

    def do_action(self, action : TradeAction):
        if action == TradeAction.DO_NOTHING:
            return

        if action == TradeAction.BUY and self.wallet == 0:
            raise TradeActionException("Insufficent funds")

        if action == TradeAction.SELL and self.tokens == 0:
            raise TradeActionException("Insufficent tokens")

        if action == TradeAction.BUY:
            self.tokens += self.wallet / self.current_token_price()
            self.wallet = 0
        elif action == TradeAction.SELL:
            self.wallet = self.tokens * self.current_token_price()
            self.tokens = 0
        else:
            raise TradeActionException("Unknown action", action)