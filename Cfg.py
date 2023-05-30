import pickle
class ConfigFiles:
    # Save object to a file
    @classmethod
    def save_object(self, obj, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    # Recover object from a file
    @classmethod
    def load_object(self, file_path):
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj