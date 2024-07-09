import pandas as pd

def calculate_sharpe_ratio(data, risk_free_rate):
    return (data['Actual Return '] - risk_free_rate) / data['Actual Return '].std()
