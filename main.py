"""
main.py
-------
Application entry point.
Run with: python main.py
Or:       uvicorn main:app --reload
"""
import uvicorn
from api.server import app, settings

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════╗
║       RAG AI System  v{settings.APP_VERSION}                  ║
║                                                  ║
║  🌐  API:   http://localhost:{settings.API_PORT}             ║
║  📖  Docs:  http://localhost:{settings.API_PORT}/docs         ║
║  🔍  UI:    Open frontend/index.html in browser  ║
╚══════════════════════════════════════════════════╝
    """)
    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
