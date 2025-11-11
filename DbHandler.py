import sqlite3
import json

class SqliteOperationHandler:
    """
    Handles SQLite operations for:
    - OutfitTypeTable
    - OutfitColorTable
    - ProductTable
    """

    def __init__(self, db_name="database.db"):
        self.db_name = db_name

    # INTERNAL CONNECTION HELPER
    def _connect(self):
        """Helper to create and return a connection and cursor."""
        conn = sqlite3.connect(self.db_name)
        return conn, conn.cursor()

    # DB TABLE CREATION
  
    def create_tables(self):
        conn, cursor = self._connect()
        print("Table creation starts...")

        tables = [
            """
            CREATE TABLE IF NOT EXISTS OUTFITTYPE (
                SEASON TEXT,
                GROUPP TEXT,
                TYPE TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS OUTFITCOLORTABLE (
                SKINTONE TEXT,
                OUTFITCOLOR TEXT
            )
            """,

            """
            CREATE TABLE IF NOT EXISTS PRODUCT (
                OUTFITTYPE TEXT,
                OUTFITCOLOR TEXT,
                PRODUCTIMAGES TEXT
            )
            """
        ]

        for table in tables:
            cursor.execute(table)
        conn.commit()
        conn.close()
        print("Table creation complete ")

    # Fetch method  

    def fetchOutfitType(self, season: str = "", group: str = ""):
        conn, cursor = self._connect()
        query = "SELECT TYPE FROM OUTFITTYPE WHERE SEASON = ? AND GROUPP = ?"
        cursor.execute(query, (season, group))
        row = cursor.fetchone()
        conn.close()

        if row:
            try:
                type_dict = json.loads(row[0])
                return {"outfittype": type_dict.get("type", [])}, True
            except json.JSONDecodeError:
                return {"outfittype": row[0]}, True
        else:
            return {"message": "no data found"}, False

    def fetchOutFitColor(self, skintoneColor: str = ""):
        conn, cursor = self._connect()
        query = "SELECT OUTFITCOLOR FROM OUTFITCOLORTABLE WHERE SKINTONE = ?"
        cursor.execute(query, (skintoneColor,))
        row = cursor.fetchone()
        conn.close()

        if row:
            try:
                color_dict = json.loads(row[0])
                return {"outfitcolor": color_dict.get("colors", [])}, True
            except json.JSONDecodeError:
                return {"outfitcolor": row[0]}, True
        else:
            return {"message": "no data found"}, False

    def fetchProduct(self, outfitcolor: str = "", outfittype: str = ""):
        conn, cursor = self._connect()
        query = """
            SELECT PRODUCTIMAGES 
            FROM PRODUCT 
            WHERE OUTFITCOLOR = ? AND OUTFITTYPE = ?
            LIMIT 3
        """
        cursor.execute(query, (outfitcolor, outfittype))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"message": "no data found"}, False

        product_images = []
        for row in rows:
            try:
                image_dict = json.loads(row[0])
                images = image_dict.get("images", [])
                if isinstance(images, list):
                    product_images.extend(images)
                else:
                    product_images.append(images)
            except json.JSONDecodeError:
                product_images.append(row[0])

        product_images = product_images[:3]
        return {"products": product_images}, True

    # INSERT METHODS

    def AddOutfitType(self, season: str, group: str, outfittype: list):
        conn, cursor = self._connect()
        type_dict = json.dumps({"type": outfittype})
        query = "INSERT INTO OUTFITTYPE (SEASON, GROUPP, TYPE) VALUES (?, ?, ?)"
        cursor.execute(query, (season, group, type_dict))
        conn.commit()
        conn.close()
        print(f" Added OutfitType for {season} - {group}")

    def AddOutfitColor(self, skintone: str, outfitcolor: list):
        conn, cursor = self._connect()
        color_dict = json.dumps({"colors": outfitcolor})
        query = "INSERT INTO OUTFITCOLORTABLE (SKINTONE, OUTFITCOLOR) VALUES (?, ?)"
        cursor.execute(query, (skintone, color_dict))
        conn.commit()
        conn.close()
        print(f" Added OutfitColor for skintone '{skintone}'")

    def AddProduct(self, outfitColor: str, outfitType: str, images: list):
        conn, cursor = self._connect()
        image_dict = json.dumps({"images": images})
        query = "INSERT INTO PRODUCT (OUTFITCOLOR, OUTFITTYPE, PRODUCTIMAGES) VALUES (?, ?, ?)"
        cursor.execute(query, (outfitColor, outfitType, image_dict))
        conn.commit()
        conn.close()
        print(f" Added Product for color '{outfitColor}' & type '{outfitType}'")

