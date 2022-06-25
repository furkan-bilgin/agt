from asyncio import base_events
from concurrent.futures import process
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
from ta.utils import dropna
from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from glob import glob

READ_DIR = sys.argv[1]
WRITE_DIR = sys.argv[2]

def process_csv(path, write_path):
    df = pd.read_csv(path)\
        .drop(columns=['Unnamed: 0'], errors='ignore')

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = dropna(df)

    macd = MACD(close=df['close'])
    rsi = RSIIndicator(close=df['close'])
    awesome = AwesomeOscillatorIndicator(high=df['high'], low=df['low'])
    stochastic_rsi = StochRSIIndicator(close=df['close'])
    #ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
    bollinger = BollingerBands(close=df["close"])

    df['i_macd'] = macd.macd_diff()
    df['i_rsi'] = rsi.rsi()
    df['i_awesome'] = awesome.awesome_oscillator()
    df['i_stochastic_rsi_d'] = stochastic_rsi.stochrsi_d()
    df['i_stochastic_rsi_k'] = stochastic_rsi.stochrsi_k()
    df['i_bollinger_h'] = bollinger.bollinger_hband_indicator()
    df['i_bollinger_l'] = bollinger.bollinger_lband_indicator()
    df.to_csv(write_path)


to_process = glob(os.path.join(READ_DIR, "*"))    

for file in to_process:
    base_name = os.path.basename(file)
    read_path = file
    write_path = os.path.join(WRITE_DIR, base_name)

    print(f'Processing {base_name}...')
    process_csv(read_path, write_path)