import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/")[0]


class MyPath(object):
    """
    Dataset root configuration.
    """

    @staticmethod
    def db_root_dir(database=""):
        db_root = os.environ.get("DATA_ROOT", "../data")
        db_names = {"PASCALContext", "NYUDv2", "imagenet"}

        if database in db_names:
            return os.path.join(db_root, database)
        elif not database:
            return db_root
        else:
            raise NotImplementedError
