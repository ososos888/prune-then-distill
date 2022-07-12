from runner.runner import run
from utils.platforms import Platform


if __name__ == "__main__":
    # Basic setup
    platform = Platform()

    platform.logger.logger.info("\nInitialization...\n")
    platform.logger.logger.info(f"The experimental results are stored in result/data/{platform.opt.experiment_name}.\n")

    run(platform)

    platform.logger.logger.info("Experiment Complete.")
