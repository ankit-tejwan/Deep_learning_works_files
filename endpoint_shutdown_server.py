# server shutdown fastapi

from fastapi import FastAPI, HTTPException,BackgroundTasks
import asyncio
import signal

shutdown_event = asyncio.Event()

@app.get("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    shutdown_event.set()
    background_tasks.add_task(log_shutdown_event)  # Optional: Log the shutdown event
    # Trigger the actual shutdown of the server after a small delay
    asyncio.create_task(shutdown_server())
    return JSONResponse(
        status_code=200,
        content={"message": "Server is shutting down"}
    )

async def log_shutdown_event():
    # Here you can implement logging logic, e.g., writing to a file or database
    print("Shutdown event triggered.")

async def shutdown_server():
    # This will trigger a shutdown by sending a signal to stop the server
    await asyncio.sleep(1)  # Give time for logging, if needed
    os.kill(os.getpid(), signal.SIGINT)  # Sends SIGINT to the process, stopping the server

