-- Monthly price and volatility by mandi
SELECT
    market,
    commodity,
    strftime('%Y-%m', arrival_date) AS month,
    COUNT(*) AS observations,
    AVG(modal_price) AS mean_modal_price,
    MIN(modal_price) AS minimum_modal_price,
    MAX(modal_price) AS maximum_modal_price
FROM mandi_prices
GROUP BY market, commodity, strftime('%Y-%m', arrival_date)
ORDER BY month, commodity, market;

-- Markets with the largest average price spread
SELECT
    state,
    market,
    AVG(max_price - min_price) AS average_price_spread,
    COUNT(*) AS observations
FROM mandi_prices
GROUP BY state, market
HAVING COUNT(*) >= 30
ORDER BY average_price_spread DESC;

-- Rolling seven-observation average using a SQL window function
SELECT
    arrival_date,
    market,
    commodity,
    modal_price,
    AVG(modal_price) OVER (
        PARTITION BY market, commodity
        ORDER BY arrival_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_7_observations
FROM mandi_prices;

