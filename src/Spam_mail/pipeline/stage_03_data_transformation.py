from Spam_mail.config.configuration import *
from Spam_mail.components.Data_Transformation import *
from Spam_mail import logger
STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                # config = ConfigurationManager()
                # data_transformation_config = config.get_data_transformation_config()
                # data_transformation = DataTransformation(config=data_transformation_config)
                # res=data_transformation.Data_transformation()
                # return res
            
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                res=data_transformation.Data_transformation()
                return res
            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)





if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
