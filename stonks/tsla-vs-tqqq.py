import yfinance as yf
from datetime import datetime, timedelta


def get_return(start_date, stonk_a, stonk_b, normalized_to_start_price=False):
    cash = 10000
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    a_qty = cash / 2 / stonk_a.loc[start_date]['Close'] 
    b_qty = cash / 2 / stonk_b.loc[start_date]['Close']

    for day in stonk_a.index:
        # print(day, start_date)
        if day <= start_date:
            # print(day, start_date, 'skipped')
            continue

        cash = a_qty * stonk_a.loc[day]['Close'] + b_qty * stonk_b.loc[day]['Close']
        a_qty = cash / 2 / stonk_a.loc[day]['Close'] 
        b_qty = cash / 2 / stonk_b.loc[day]['Close']

    # print('===================')
    if normalized_to_start_price:
        cash = a_qty * stonk_a.loc[start_date]['Close'] + b_qty * stonk_b.loc[start_date]['Close']
    return cash

def get_shares(start_date, stonk_a, stonk_b, all_in_stonk_a=False):
    cash = 10000
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    stonk_a_start_price = stonk_a.loc[start_date]['Close']
    a_qty = cash / 2 / stonk_a.loc[start_date]['Close'] 
    b_qty = cash / 2 / stonk_b.loc[start_date]['Close']
    initial_a_qty = a_qty
    initial_all_in_a_qty = cash / stonk_a.loc[start_date]['Close']
    initial_b_qty = b_qty

    for day in stonk_a.index:
        if day <= start_date:
            continue

        cash = a_qty * stonk_a.loc[day]['Close'] + b_qty * stonk_b.loc[day]['Close']
        a_qty = cash / 2 / stonk_a.loc[day]['Close'] 
        b_qty = cash / 2 / stonk_b.loc[day]['Close']

    if all_in_stonk_a:
        stonk_a_end_price = stonk_a.loc[stonk_a.index[-1]]['Close']
        return cash / stonk_a_end_price / initial_all_in_a_qty, 0

    return a_qty / initial_a_qty, b_qty / initial_b_qty

def buy_and_hold(start_date, stonk):
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    cash = 10000
    stonk_qty = cash / stonk.loc[start_date]['Close']

    cash = stonk_qty * stonk.loc[stonk.index[-1]]['Close']
    return cash



def main():
    # Define the date range
    end = datetime.now()
    start = end - timedelta(days=2*365)

    # Fetch the data
    tsla = yf.download('TSLA', start=start, end=end)
    tqqq = yf.download('TQQQ', start=start, end=end)

    # Save the data to CSV files
    tsla.to_csv('TSLA_5_years.csv')
    tqqq.to_csv('TQQQ_5_years.csv')

    for day in tsla.index:
        cash = get_return(day, tsla, tqqq)
        # print(day, cash)

        tsla_allin, _ = get_shares(day, tsla, tqqq, True)
        tqqq_allin, _ = get_shares(day, tqqq, tsla, True)

        hodl_tsla = buy_and_hold(day, tsla)
        hodl_tqqq = buy_and_hold(day, tqqq)
        # print(day, cash / hodl_tsla, cash / hodl_tqqq)

        print(day, tsla_allin, tqqq_allin, cash, hodl_tsla, hodl_tqqq)

    # get_return('2020-01-03', tsla, tqqq)



if __name__ == '__main__':
    main()