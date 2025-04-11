-- Data Analytics Bootcamp Project: Data Cleaning
--- CONTENTS
	--- PART 1: DAILY MARKET DATA
	--- PART 2: INTERPOLATE MONTHLY FREQUENCY AND CREATE AN ECONOMY DATASET, THEN EXPORT TO CSV AND FINISH INTERPOLATION IN EXCEL
	--- PART 3: MONTHLY DATA
	--- PART 4: QUARTERLY DATA
	--- PART 5: QA
	--- PART 6: EXPORT TABLES


-- PART 1: DAILY MARKET DATA
--- stock market, VIX, gold, S&P, Bitocin
CREATE TABLE daily_market_data AS --- this will be used for daily stock market forecasting
SELECT 
	st.date_ymd, 
	st."Technology" AS technology, 
	st."Financials" AS financials, 
	st."Healthcare" AS healthcare,
	st."Consumer Discretionary" AS consumer_discretionary,
	st."Utilities" AS utilities,
	st."Industrials" AS industrials,
	st."Consumer Staples" AS consumer_staples,
	vix."Close" AS vix_close,
	gp."USD" AS gold_us_price, --- NULLs in gold data
	sp."Close" AS sandp_close,
	bc."Close" AS bitcoin_close
FROM sector_stock_data_daily_new AS st
LEFT JOIN vix_data_daily_new AS vix
	ON st.date_ymd = vix.date_ymd
LEFT JOIN gold_prices AS gp
	ON st.date_ymd = gp.date_ymd
LEFT JOIN sp500_data_daily_new AS sp
	ON st.date_ymd = sp.date_ymd
LEFT JOIN bitcoin_data_btc_usd_from AS bc
	ON st.date_ymd = bc.date_ymd; 	

SELECT * FROM daily_market_data ORDER BY date_ymd



-- PART 2: INTERPOLATE MONTHLY FREQUENCY AND CREATE AN ECONOMY DATASET, THEN EXPORT TO CSV AND FINISH INTERPOLATION IN EXCEL
--- Create monthly frequency
WITH months AS (
    SELECT date_trunc('month', d)::DATE AS month_date
    FROM GENERATE_SERIES(
        (SELECT MIN(date) FROM gdp_cpi_data_quarterly), 
        (SELECT MAX(date) FROM gdp_cpi_data_quarterly), 
        INTERVAL '1 month'
    ) d
)
SELECT m.month_date, gcq.*
FROM months m
LEFT JOIN gdp_cpi_data_quarterly AS gcq
ON DATE_TRUNC('quarter', m.month_date) = gcq.date;



--- Interpolate monthly data to economic data
CREATE TABLE economy_monthly_data_new AS 
WITH monthly_data AS (
    -- Generate all months between the min and max date
    SELECT date_trunc('month', d)::DATE AS month_date
    FROM GENERATE_SERIES(
        (SELECT MIN(date) FROM gdp_cpi_data_quarterly), 
        (SELECT MAX(date) FROM gdp_cpi_data_quarterly), 
        INTERVAL '1 month'
    ) d
),
quarterly_joined AS (
    -- Attach quarterly data to months (some will be NULL)
    SELECT 
        m.month_date, 
        gcq.date AS quarter_date, 
        gcq.rgdp2017, 
        gcq.cpi
    FROM monthly_data AS m
    LEFT JOIN gdp_cpi_data_quarterly AS gcq
    ON m.month_date = gcq.date
),
quarterly_values AS (
    -- Identify previous and next quarter values for interpolation
    SELECT 
        month_date,
        rgdp2017,
        cpi,
        -- Previous quarter values
        FIRST_VALUE(rgdp2017) OVER (PARTITION BY quarter_date ORDER BY month_date ASC 
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_gdp,
        FIRST_VALUE(cpi) OVER (PARTITION BY quarter_date ORDER BY month_date ASC 
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_cpi,
        -- Next quarter values
        FIRST_VALUE(rgdp2017) OVER (PARTITION BY quarter_date ORDER BY month_date DESC 
                                    ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_gdp,
        FIRST_VALUE(cpi) OVER (PARTITION BY quarter_date ORDER BY month_date DESC 
                               ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_cpi
    FROM quarterly_joined
),
interpolated AS (
    SELECT 
        month_date,

        -- Interpolated GDP
        CASE 
            WHEN rgdp2017 IS NOT NULL THEN rgdp2017 -- Keep original values
            ELSE prev_gdp + ((next_gdp - prev_gdp) / 3) * ((EXTRACT(MONTH FROM month_date) - 1) % 3 + 1)
        END AS interpolated_gdp,

        -- Interpolated CPI
        CASE 
            WHEN cpi IS NOT NULL THEN cpi
            ELSE prev_cpi + ((next_cpi - prev_cpi) / 3) * ((EXTRACT(MONTH FROM month_date) - 1) % 3 + 1)
        END AS interpolated_cpi
    FROM quarterly_values
)
SELECT * FROM interpolated;




-- PART 3: MONTHLY DATA
--- Interpolated GDP and CPI monthly data with IRs
CREATE TABLE regression_data AS
SELECT edmi.month_date, edmi.interpolated_gdp, edmi.interpolated_cpi, irus."FEDFUNDS" AS interest_rate_us
FROM economy_data_monthly_interpolated AS edmi
LEFT JOIN ir_us_new AS irus
ON edmi.month_date = irus.date_ymd;

SELECT * FROM regression_data


--- Economy data
CREATE TABLE economy_data_final AS
SELECT 
    rd.*,
    ur.unemployment_rate
FROM regression_data AS rd
LEFT JOIN unemployment_rate_us_monthly_new AS ur 
	ON rd.month_date = ur.date_ymd
ORDER BY month_date; --- distinct dates


SELECT * FROM economy_data_final ORDER BY month_date --- economic data from 1956, no exchange rates. This is the start of the month, e.g. 1956-01-01


--- GDP growth rate here (monthly), this makes joins less messy later
CREATE TABLE economy_data_with_gdpgrowthperm AS 
WITH gdp_growth_m AS (
    SELECT
    	month_date,
    	interpolated_gdp,
        LAG(interpolated_gdp) OVER (ORDER BY month_date) AS previous_gdp_value,
        ((interpolated_gdp - LAG(interpolated_gdp) OVER (ORDER BY month_date)) / LAG(interpolated_gdp) OVER (ORDER BY month_date)) * 100 AS gdp_growth_rate
    	FROM economy_data_final
    	)
	SELECT 
	edf.*, ggm.gdp_growth_rate   
	FROM economy_data_final AS edf
	LEFT JOIN gdp_growth_m AS ggm
		ON edf.month_date = ggm.month_date
	WHERE previous_gdp_value IS NOT NULL
	ORDER BY month_date;

SELECT * FROM economy_data_with_gdpgrowthperm --- economy data, with GDP growth rate, from 1956, distinct dates




--- 1) CONNECT TO ERs, 2) CONNECT TO DAILY AGGREGATED STOCK MARKET DATA, 3) REPEAT PROCESS FOR QUARTERLY DATA
SELECT * FROM economy_data_exchangerates_syncronisedtime

---1) CONNECT TO ERs
CREATE TABLE economy_data_all_monthly AS
SELECT edg.*, edes.euro_dollar
FROM economy_data_with_gdpgrowthperm AS edg
LEFT JOIN economy_data_exchangerates_syncronisedtime AS edes
	ON edg.month_date = edes.month_date
ORDER BY month_date; --- economy data (including GDP growth rate) with ERs, from 1956, monthly, distinct dates

SELECT * FROM economy_data_all_monthly


--- 2) CONNECT TO DAILY AGGREGATED STOCK MARKET DATA
CREATE TABLE economy_stockmarket_monthly AS
WITH aggregated AS (
	SELECT
		DATE_TRUNC('month', date_ymd) AS date,
		AVG(technology) AS avg_technology,
		AVG(financials) AS avg_financials,
		AVG(healthcare) AS avg_healthcare,
		AVG(consumer_discretionary) AS avg_consumer_discretionary,
		AVG(utilities) AS avg_utilities,
		AVG(industrials) AS avg_industrials,
		AVG(consumer_staples) AS avg_consumer_staples,
		AVG(vix_close) AS avg_vix_close,
		AVG(gold_us_price) AS avg_gold_us_price,
		AVG(sandp_close) AS avg_sandp_close,
		AVG(bitcoin_close) AS avg_bitcoin_close
	FROM daily_market_data
	GROUP BY DATE_TRUNC('month', date_ymd)
	)
SELECT edam.*, a.*
FROM economy_data_all_monthly AS edam
LEFT JOIN aggregated AS a
	ON edam.month_date  = a.date
ORDER BY month_date;

SELECT * FROM economy_stockmarket_monthly --- economy (including GDP growth rate) and stock market data, monthly, from 1956

--- Extra: data from 1999 (meditum term), where there is all data which is not null (except for Bitcoin)
CREATE TABLE economy_sm_monthly_mt AS 
SELECT * FROM economy_stockmarket_monthly
WHERE euro_dollar IS NOT NULL 

SELECT * FROM economy_sm_monthly_mt

--- tables that I want to export:
--- 1) economy_stockmarket_monthly,
--- 2) economy_sm_monthly_mt









-- PART 4: QUARTERLY DATA
--- GDP and CPI data: filtered to exclude CPI NULLs
CREATE TABLE gdp_cpi_data_quarterly AS
SELECT rg.date_ymd AS date, rg."Real GDP ($billion 2017)" AS rgdp2017, cqdfn."CPI" AS cpi 
FROM real_gdp AS rg 
LEFT JOIN cpi_quarterly_data_fred_new AS cqdfn 
ON rg.date_ymd = cqdfn.date_ymd;


--- 1) quarterly data of GDP, CPI, unemployment rate and interest rates from 1956
CREATE TABLE economy_data_longterm_quarterly AS
	SELECT gcd.*, ur.unemployment_rate, ir."FEDFUNDS" AS interest_rate_us
	FROM gdp_cpi_data_quarterly AS gcd 
	LEFT JOIN unemployment_rate_us_monthly_new AS ur 
		ON gcd.date = ur.date_ymd
	LEFT JOIN ir_us_new AS ir
    ON DATE_TRUNC('month', gcd.date) = DATE_TRUNC('month', ir.date_ymd) --- quarterly from 1956
    ORDER BY date

    

--- GDP growth rate here
CREATE TABLE economy_data_with_gdpgrowthperq AS 
WITH gdp_growth_q AS (
    SELECT
    	date,
    	rgdp2017,
        LAG(rgdp2017) OVER (ORDER BY date) AS previous_gdp_value,
        ((rgdp2017 - LAG(rgdp2017) OVER (ORDER BY date)) / LAG(rgdp2017) OVER (ORDER BY date)) * 100 AS gdp_growth_rate
    	FROM economy_data_longterm_quarterly
    	)
	SELECT 
	edl.*, ggq.gdp_growth_rate   
	FROM economy_data_longterm_quarterly AS edl
	LEFT JOIN gdp_growth_q AS ggq
		ON edl.date = ggq.date
	WHERE previous_gdp_value IS NOT NULL
	ORDER BY date;

SELECT * FROM economy_data_with_gdpgrowthperq --- economy data, with GDP growth rate, from 1956, distinct dates


--- Join exchange rates
SELECT * FROM quarterly_data_all

CREATE TABLE economy_data_quarterly AS
SELECT edgq.*, q.euro_dollar
FROM economy_data_with_gdpgrowthperq AS edgq
LEFT JOIN quarterly_data_all AS q
	ON edgq.date = q.date
ORDER BY date;
	
SELECT * FROM economy_data_quarterly


--- Aggregate stock market data by quarter
CREATE TABLE stockmarket_quarterly AS 
	SELECT
		DATE_TRUNC('quarter', date_ymd) AS quarter_date,
		AVG(technology) AS avg_technology,
		AVG(financials) AS avg_financials,
		AVG(healthcare) AS avg_healthcare,
		AVG(consumer_discretionary) AS avg_consumer_discretionary,
		AVG(utilities) AS avg_utilities,
		AVG(industrials) AS avg_industrials,
		AVG(consumer_staples) AS avg_consumer_staples,
		AVG(vix_close) AS avg_vix_close,
		AVG(gold_us_price) AS avg_gold_us_price,
		AVG(sandp_close) AS avg_sandp_close,
		AVG(bitcoin_close) AS avg_bitcoin_close
	FROM daily_market_data
	GROUP BY DATE_TRUNC('quarter', date_ymd)
ORDER BY quarter_date;



CREATE TABLE quarterly_data_all AS
SELECT edq.*, sq.*
FROM economy_data_quarterly AS edq
LEFT JOIN stockmarket_quarterly AS sq 
	ON DATE_TRUNC('quarter', edq.date) = sq.quarter_date
ORDER BY edq.date; --- 

SELECT * FROM quarterly_data_all ORDER BY date --- problem is that only 1999 data is duplicated


--- This method is giving me some duplicate dates, therefore use SELECT DISTINCT ON:
CREATE TABLE economy_stockmarket_quarterly AS
WITH aggregated AS (
	SELECT
		DATE_TRUNC('quarter', date_ymd) AS quarter_date,
		AVG(technology) AS avg_technology,
		AVG(financials) AS avg_financials,
		AVG(healthcare) AS avg_healthcare,
		AVG(consumer_discretionary) AS avg_consumer_discretionary,
		AVG(utilities) AS avg_utilities,
		AVG(industrials) AS avg_industrials,
		AVG(consumer_staples) AS avg_consumer_staples,
		AVG(vix_close) AS avg_vix_close,
		AVG(gold_us_price) AS avg_gold_us_price,
		AVG(sandp_close) AS avg_sandp_close,
		AVG(bitcoin_close) AS avg_bitcoin_close
	FROM daily_market_data
	GROUP BY DATE_TRUNC('quarter', date_ymd)
	)
SELECT DISTINCT ON (edq.date)
	edq.*, 
	a.*
FROM economy_data_quarterly AS edq
LEFT JOIN aggregated AS a
	ON edq.date  = a.quarter_date
ORDER BY date;

SELECT * FROM economy_stockmarket_quarterly

--- Extra: quarterly data medium term
CREATE TABLE economy_sm_quarterly_mt AS
SELECT * FROM economy_stockmarket_quarterly
WHERE euro_dollar IS NOT NULL

SELECT * FROM economy_sm_quarterly_mt


--- Tables I will EXPORT
---1) economy_stockmarket_quarterly
---2) economy_sm_quarterly_mt





-- PART 5: QA
--- daily_market_data
SELECT date_ymd, COUNT(*)
FROM daily_market_data
GROUP BY date_ymd
HAVING COUNT(*) > 1
ORDER BY date_ymd; --- daily_market_data has distinct dates

--- economy_stockmarket_monthly
SELECT month_date, COUNT(*)
FROM economy_stockmarket_monthly
GROUP BY month_date
HAVING COUNT(*) > 1
ORDER BY month_date; --- economy_stockmarket_monthly has distinct dates

--- economy_monthly_mt
SELECT month_date, COUNT(*)
FROM economy_stockmarket_monthly
GROUP BY month_date
HAVING COUNT(*) > 1
ORDER BY month_date; --- economy_monthly_mt has distinct dates

--- economy_data_longterm_quarterly
SELECT date, COUNT(*)
FROM economy_data_longterm_quarterly
GROUP BY date
HAVING COUNT(*) > 1
ORDER BY date; --- economy_data_longterm_quarterly has distinct dates

--- economy_data_with_gdpgrowthperq
SELECT date, COUNT(*)
FROM economy_data_with_gdpgrowthperq
GROUP BY date
HAVING COUNT(*) > 1
ORDER BY date; --- economy_data_with_gdpgrowthperq has distinct dates

--- quarterly_data_all
SELECT date, COUNT(*)
FROM quarterly_data_all
GROUP BY date
HAVING COUNT(*) > 1
ORDER BY date; --- quarterly_data_all DOES NOT HAVE distinct dates

--- stockmarket_quarterly
SELECT quarter_date, COUNT(*)
FROM stockmarket_quarterly
GROUP BY quarter_date
HAVING COUNT(*) > 1
ORDER BY quarter_date; --- stockmarket_quarterly has distinct dates

--- economy_stockmarket_quarterly
SELECT date, COUNT(*)
FROM economy_stockmarket_quarterly
GROUP BY date
HAVING COUNT(*) > 1
ORDER BY date; --- economy_stockmarket_quarterly has distinct dates

--- economy_sm_quarterly_mt
SELECT date, COUNT(*)
FROM economy_sm_quarterly_mt
GROUP BY date
HAVING COUNT(*) > 1
ORDER BY date; --- economy_sm_quarterly_mt has distinct dates





-- PART 6: EXPORT TABLES
--- 1) daily_market_data
SELECT * FROM daily_market_data

--- 2) economy_stockmarket_monthly
SELECT * FROM economy_stockmarket_monthly

--- 3) economy_sm_monthly_mt
SELECT * FROM economy_sm_monthly_mt

--- 4) economy_stockmarket_quarterly
SELECT * FROM economy_stockmarket_quarterly
ORDER BY date DESC

--- 5) economy_sm_quarterly_mt
SELECT * FROM economy_sm_quarterly_mt
