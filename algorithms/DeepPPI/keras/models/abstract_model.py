from abc import ABC, abstractmethod
 
class AbstractModel(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_model(self):
        pass
