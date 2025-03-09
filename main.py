import logging
import argparse
from src.Sign_Language_Classification.pipeline.train_pipeline import TrainPipeline
from src.Sign_Language_Classification.pipeline.inference_pipeline import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:"
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sign Language Classification")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="inference",
        help="Mode to run: 'train' to train a new model, 'inference' for real-time inference (default)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "train":
            logging.info("Running in training mode...")
            pipeline = TrainPipeline()
            pipeline.run_pipeline()
        else:  # inference mode
            logging.info("Running in inference mode...")
            pipeline = InferencePipeline()
            pipeline.run_pipeline()
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == "__main__":
    main()
