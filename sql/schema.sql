CREATE TABLE IF NOT EXISTS mandi_prices (
    state TEXT NOT NULL,
    district TEXT NOT NULL,
    market TEXT NOT NULL,
    commodity TEXT NOT NULL,
    variety TEXT NOT NULL,
    grade TEXT,
    arrival_date DATE NOT NULL,
    min_price REAL NOT NULL CHECK (min_price > 0),
    max_price REAL NOT NULL CHECK (max_price > 0),
    modal_price REAL NOT NULL CHECK (modal_price > 0),
    arrival_quantity REAL,
    rainfall_mm REAL,
    temp_max_c REAL,
    latitude REAL,
    longitude REAL,
    PRIMARY KEY (state, district, market, commodity, variety, arrival_date),
    CHECK (min_price <= modal_price AND modal_price <= max_price)
);

CREATE INDEX IF NOT EXISTS idx_mandi_lookup
ON mandi_prices (commodity, market, arrival_date);

