import asyncio
import websockets
import json
import csv
from datetime import datetime

CSV_FILE = "bsevents_data.csv"

# Create CSV file if not exists
def create_csv():
    try:
        with open(CSV_FILE, "x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "song_time",
                "score",
                "accuracy",
                "combo",
                "missCount",
                "health"
            ])
    except FileExistsError:
        pass


async def log_data():
    uri = "ws://localhost:2947/socket"

    while True:
        try:
            print(f"Connecting to {uri}...")
            async with websockets.connect(uri) as ws:
                print("Connected!")

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    # Only process score events
                    if data.get("_type") == "event" and data.get("_event") == "score":
                        ev = data["scoreEvent"]

                        timestamp = datetime.utcnow().isoformat()

                        row = [
                            timestamp,
                            ev.get("time"),
                            ev.get("score"),
                            ev.get("accuracy"),
                            ev.get("combo"),
                            ev.get("missCount"),
                            ev.get("currentHealth")
                        ]

                        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(row)

                        print(f"[{timestamp}] t={ev.get('time')} score={ev.get('score')} "
                              f"acc={ev.get('accuracy'):.3f} combo={ev.get('combo')} miss={ev.get('missCount')}")

        except Exception as e:
            print("Connection lost. Retrying in 2 seconds...", e)
            await asyncio.sleep(2)


if __name__ == "__main__":
    create_csv()
    asyncio.run(log_data())
