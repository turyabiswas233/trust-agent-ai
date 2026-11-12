import argparse
import random
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "store.db"
CUSTOMER_CITIES = (
    "Dhaka",
    "Chattogram",
    "Khulna",
    "Rajshahi",
    "Sylhet",
    "Barishal",
    "Rangpur",
    "Cumilla",
)
PRODUCT_CATALOG = (
    ("Wireless Mouse", "Electronics", 12.99, 34.99),
    ("Mechanical Keyboard", "Electronics", 39.99, 119.99),
    ("USB-C Hub", "Electronics", 18.99, 69.99),
    ("Desk Lamp", "Home", 14.99, 54.99),
    ("Water Bottle", "Lifestyle", 8.99, 29.99),
    ("Notebook Set", "Stationery", 4.99, 19.99),
    ("Running Shoes", "Sports", 35.99, 129.99),
    ("Yoga Mat", "Sports", 12.99, 49.99),
    ("Coffee Grinder", "Home", 24.99, 89.99),
    ("Travel Backpack", "Lifestyle", 29.99, 109.99),
    ("Bluetooth Speaker", "Electronics", 22.99, 99.99),
    ("Cookware Set", "Home", 39.99, 159.99),
    ("T-Shirt", "Apparel", 9.99, 34.99),
    ("Hoodie", "Apparel", 24.99, 79.99),
    ("Pen Set", "Stationery", 3.99, 14.99),
    ("Fitness Band", "Electronics", 19.99, 84.99),
    ("Board Game", "Lifestyle", 14.99, 59.99),
    ("Camping Lantern", "Sports", 16.99, 64.99),
    ("Wall Clock", "Home", 11.99, 44.99),
    ("Sunglasses", "Apparel", 14.99, 69.99),
)
FIRST_NAMES = (
    "Amina", "Arif", "Farah", "Hasan", "Ishita", "Jamil", "Nadia", "Rafi",
    "Sadia", "Tanvir", "Maya", "Omar", "Priya", "Sami", "Tania", "Zubair",
)
LAST_NAMES = (
    "Ahmed", "Akter", "Chowdhury", "Das", "Hossain", "Islam", "Khan", "Rahman",
    "Sultana", "Uddin",
)


def build_database(
    db_path: str | Path = DEFAULT_DB_PATH, seed: int = 20251112
) -> Path:
    """Create the same four-table sample store database for a given seed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    base_date = date(2024, 1, 1)

    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS order_items;
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS customers;

            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                signup_date TEXT NOT NULL
            );
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL CHECK (price >= 0)
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                order_date TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL CHECK (unit_price >= 0)
            );
            """
        )

        customers = []
        for customer_id in range(1, 41):
            name = (
                f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            )
            signup = base_date + timedelta(days=rng.randrange(0, 900))
            customers.append(
                (customer_id, name, rng.choice(CUSTOMER_CITIES), signup.isoformat())
            )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)", customers
        )

        products = []
        for product_id, (name, category, low, high) in enumerate(
            PRODUCT_CATALOG, start=1
        ):
            cents = rng.randint(round(low * 100), round(high * 100))
            products.append((product_id, name, category, cents / 100))
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)", products
        )

        orders = []
        items = []
        statuses = ("delivered", "delivered", "delivered", "shipped", "cancelled")
        item_id = 1
        for order_id in range(1, 121):
            order_date = base_date + timedelta(days=rng.randrange(0, 1000))
            orders.append(
                (
                    order_id,
                    rng.randint(1, len(customers)),
                    order_date.isoformat(),
                    rng.choice(statuses),
                )
            )
            chosen_products = rng.sample(products, rng.randint(2, 4))
            for product in chosen_products:
                items.append(
                    (
                        item_id,
                        order_id,
                        product[0],
                        rng.randint(1, 5),
                        product[3],
                    )
                )
                item_id += 1

        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
        connection.executemany(
            "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic sample store database.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--seed", type=int, default=20251112)
    args = parser.parse_args()
    path = build_database(args.db, seed=args.seed)
    with closing(sqlite3.connect(path)) as connection:
        row_count = connection.execute(
            "SELECT (SELECT COUNT(*) FROM customers) + "
            "(SELECT COUNT(*) FROM products) + "
            "(SELECT COUNT(*) FROM orders) + "
            "(SELECT COUNT(*) FROM order_items)"
        ).fetchone()[0]
    print(f"Built {path} with 4 tables and {row_count} total rows.")


if __name__ == "__main__":
    main()
