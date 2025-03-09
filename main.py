import logging
from src.Sign_Language_Classification.pipeline.train_pipeline import TrainPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:"
)

def main():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == "__main__":
    main()
