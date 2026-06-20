"""
Face Detection API Server Launcher
=================================
Easy startup script for the REST API server.
"""


def check_dependencies():
    """Check if required dependencies are installed."""
    required_modules = ["flask", "werkzeug", "requests"]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print("❌ Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n📦 Install missing modules:")
        print("   pip install -r api/requirements.txt")
        return False

    return True


def check_triton_server():
    """Check if Triton server is accessible."""
    try:
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient(url="localhost:8001")
        if client.is_server_ready():
            print("✅ Triton server is ready")
            return True
        else:
            print("⚠️  Triton server not ready")
            return False
    except Exception as e:
        print(f"❌ Triton server connection failed: {e}")
        print("🔧 Start Triton server first:")
        print("   docker run --gpus all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 \\")
        print("       -v$(pwd)/triton_face_detection/model_repository:/models \\")
        print("       nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models")
        return False


def main():
    """Main startup function."""
    print("🚀 Face Detection & ArcFace Embedding API Server")
    print("=" * 50)

    # Check dependencies
    print("📋 Checking dependencies...")
    if not check_dependencies():
        return

    # Check Triton server
    print("🔍 Checking Triton server...")
    triton_ready = check_triton_server()

    # Setup directories
    print("📁 Setting up directories...")
    import config

    config.ensure_directories()
    print(f"   - Upload folder: {config.UPLOAD_FOLDER}")
    print(f"   - Temp folder: {config.TEMP_FOLDER}")
    print(f"   - Output folder: {config.OUTPUT_FOLDER}")

    # Triton warning if needed
    if not triton_ready:
        print("\n⚠️  WARNING: Triton server not ready. Face detection will fail!")
        print("   Continue anyway? (y/n): ", end="")
        if input().lower() != "y":
            return

    print("\n" + "=" * 50)
    print("🎯 Server ready for requests!")
    print("📖 Test with: curl http://localhost:5000/api/v1/health")
    print("🛑 Stop with: Ctrl+C")
    print("=" * 50)

    # Start server
    try:
        print("📥 Importing Flask app...")
        from app import app

        print("✅ Flask app imported successfully")

        print(f"🚀 Starting Flask server on {config.SERVER_HOST}:{config.SERVER_PORT}")
        app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=config.DEBUG_MODE, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
