import duckdb

# Connect to the database
conn = duckdb.connect("salesdata.db")

# Run queries
conn.execute("SELECT * FROM salesperson").df()
conn.execute("SELECT * FROM orders LIMIT 10").df()
conn.execute("SELECT * FROM training").df()
conn.execute("SELECT * FROM bonus_pay").df()
conn.execute("SELECT * FROM agent_commissions LIMIT 5").df()

# Custom queries
print(conn.execute("""
SELECT 
    s.name,
    COUNT(*) as invalid_sales_count
FROM orders o
JOIN salesperson s ON o.salesperson_id = s.id
LEFT JOIN training t ON o.salesperson_id = t.salesperson_id 
    AND o.order_date BETWEEN t.start_date AND t.end_date
WHERE t.id IS NULL  -- No matching training record = invalid
GROUP BY s.name
ORDER BY invalid_sales_count DESC
LIMIT 1;

""").df())

# Close when done
conn.close()