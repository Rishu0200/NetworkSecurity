# Add these new imports at the top
from dagshub import storage
from pathlib import Path
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create required directories
Path("final_model").mkdir(exist_ok=True)
Path("prediction_output").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Modified MongoDB connection with error handling
@app.on_event("startup")
async def startup_db_client():
    global client, database, collection
    try:
        client = pymongo.MongoClient(
            os.getenv("MONGODB_URL_KEY"), 
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000
        )
        # Test the connection
        client.admin.command('ismaster')
        database = client[DATA_INGESTION_DATABASE_NAME]
        collection = database[DATA_INGESTION_COLLECTION_NAME]
        logger.info("✅ MongoDB connected successfully")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise

# Add Dagshub model loading function
def load_from_dagshub_if_needed():
    model_files = {
        "preprocessor.pkl": "path/to/preprocessor.pkl",
        "model.pkl": "path/to/model.pkl"
    }
    
    for local_file, dagshub_path in model_files.items():
        local_path = f"final_model/{local_file}"
        if not os.path.exists(local_path):
            try:
                storage.download(
                    f"dagshub://{os.getenv('DAGSHUB_USER')}/your-repo/{dagshub_path}",
                    local_path
                )
                logger.info(f"Downloaded {local_file} from Dagshub")
            except Exception as e:
                logger.error(f"Failed to download {local_file}: {e}")
                raise

# Modified predict endpoint
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Load models from Dagshub if missing locally
        load_from_dagshub_if_needed()
        
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        y_pred = network_model.predict(df)
        
        df['predicted_column'] = y_pred
        output_path = Path("prediction_output/output.csv")
        df.to_csv(output_path)
        
        return templates.TemplateResponse(
            "table.html", 
            {"request": request, "table": df.to_html(classes='table table-striped')}
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise NetworkSecurityException(e, sys)

# Modified main block
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    app_run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        reload=False
    )